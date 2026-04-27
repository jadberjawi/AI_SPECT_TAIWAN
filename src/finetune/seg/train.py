import glob
import json
import time
import yaml
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
# import wandb
import nibabel as nib

from src.backbones.resnet3d_gn import resnet18_3d_gn
from src.finetune.seg.heads import SegHeadFromResNetFeatures


# -------------------------
# Dataset (file-name based; required for CV)
# -------------------------
class PairSegDataset(Dataset):
    """
    Expects:
      images_dir/<n>.nii(.gz)
      labels_dir/<n>.nii(.gz)
    with identical filenames.

    """
    def __init__(self, images_dir: str, labels_dir: str, cfg: dict, file_names: list[str]):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.cfg = cfg

        if len(file_names) == 0:
            raise RuntimeError("Empty file list for dataset split.")

        # store full paths for speed / clarity
        self.files = [self.images_dir / fn for fn in file_names]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        img_path = self.files[idx]
        lbl_path = self.labels_dir / img_path.name
        if not lbl_path.exists():
            raise RuntimeError(f"Missing label: {lbl_path}")

        img = nib.load(str(img_path)).get_fdata().astype(np.float32)
        lbl = nib.load(str(lbl_path)).get_fdata().astype(np.float32)

        lbl = (lbl > 0.5).astype(np.float32)

        print(f"Image: {img_path.name}, img shape={img.shape} lbl shape={lbl.shape} img range=({img.min():.2f},{img.max():.2f}) lbl unique={np.unique(lbl)}")

        x = torch.from_numpy(img)[None, ...]  # (1,40,40,40)
        y = torch.from_numpy(lbl)[None, ...]
        return x, y, img_path.name


# -------------------------
# Losses / metrics
# -------------------------
class SoftDiceLoss(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        num = 2 * (probs * targets).sum(dim=(2, 3, 4))
        den = (probs + targets).sum(dim=(2, 3, 4)) + self.eps
        dice = num / den
        return 1.0 - dice.mean()


@torch.no_grad()
def metrics_binary(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> dict:
    """Per-sample metrics returned as lists (not reduced), so caller can compute mean AND std."""
    probs = torch.sigmoid(logits)
    pred = (probs > 0.5).float()

    tp = (pred * targets).sum(dim=(2, 3, 4))
    fp = (pred * (1 - targets)).sum(dim=(2, 3, 4))
    fn = ((1 - pred) * targets).sum(dim=(2, 3, 4))
    tn = ((1 - pred) * (1 - targets)).sum(dim=(2, 3, 4))

    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)

    # Volume ratio: pred_volume / gt_volume — tracks over/under-segmentation
    pred_vol = pred.sum(dim=(2, 3, 4))
    gt_vol = targets.sum(dim=(2, 3, 4))
    vol_ratio = (pred_vol + eps) / (gt_vol + eps)

    return {
        "dice": dice.cpu().tolist(),
        "iou": iou.cpu().tolist(),
        "precision": precision.cpu().tolist(),
        "recall": recall.cpu().tolist(),
        "vol_ratio": vol_ratio.cpu().tolist(),
    }


@torch.no_grad()
def boundary_f1_3d(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> list[float]:
    """
    Surface Dice / Boundary F1 approximation for 3D binary masks.
    Computes boundary by morphological erosion (3x3x3 kernel) and measures
    overlap of boundary voxels. This is a lightweight proxy used in
    nnU-Net and Cityscapes-style evaluation.
    Returns per-sample BF1 as a list.
    """
    pred = (torch.sigmoid(logits) > 0.5).float()

    # 3x3x3 uniform kernel for erosion
    kernel = torch.ones(1, 1, 3, 3, 3, device=logits.device)

    def extract_boundary(mask: torch.Tensor) -> torch.Tensor:
        # mask: (B,1,D,H,W)
        eroded = (nn.functional.conv3d(mask, kernel, padding=1) >= kernel.numel()).float()
        return mask - eroded  # boundary = original - eroded

    pred_bnd = extract_boundary(pred)
    gt_bnd = extract_boundary(targets)

    tp = (pred_bnd * gt_bnd).sum(dim=(2, 3, 4))
    fp = (pred_bnd * (1 - gt_bnd)).sum(dim=(2, 3, 4))
    fn = ((1 - pred_bnd) * gt_bnd).sum(dim=(2, 3, 4))

    bf1 = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    return bf1.cpu().tolist()


@torch.no_grad()
def compute_dead_neuron_ratio(encoder: nn.Module, head: nn.Module, dl, device, max_batches: int = 10) -> dict:
    """
    Compute fraction of ReLU/GELU outputs that are always zero across a
    subset of validation data. High ratio → representation collapse.
    """
    encoder.eval()
    head.eval()

    hooks = []
    activation_counts = {}  # layer_name -> (total_activations, zero_activations)

    def make_hook(name):
        def hook_fn(module, input, output):
            if isinstance(output, torch.Tensor):
                total = output.numel()
                zeros = (output == 0).sum().item()
                if name in activation_counts:
                    activation_counts[name] = (
                        activation_counts[name][0] + total,
                        activation_counts[name][1] + zeros,
                    )
                else:
                    activation_counts[name] = (total, zeros)
        return hook_fn

    # Hook after all ReLU/GELU layers in encoder
    for name, module in encoder.named_modules():
        if isinstance(module, (nn.ReLU, nn.GELU, nn.LeakyReLU)):
            hooks.append(module.register_forward_hook(make_hook(f"enc.{name}")))

    for name, module in head.named_modules():
        if isinstance(module, (nn.ReLU, nn.GELU, nn.LeakyReLU)):
            hooks.append(module.register_forward_hook(make_hook(f"head.{name}")))

    with torch.no_grad():
        for i, (x, y, _) in enumerate(dl):
            if i >= max_batches:
                break
            x = x.to(device, non_blocking=True)
            feat = encoder.forward_features(x)
            _ = head(feat)

    for h in hooks:
        h.remove()

    # Aggregate
    total_all = 0
    zero_all = 0
    per_layer = {}
    for name, (total, zeros) in activation_counts.items():
        ratio = zeros / max(total, 1)
        per_layer[name] = ratio
        total_all += total
        zero_all += zeros

    global_ratio = zero_all / max(total_all, 1)
    return {"dead_neuron/global_ratio": global_ratio, "dead_neuron/per_layer": per_layer}


def save_pred_example(out_dir: Path, name: str, x: torch.Tensor, y: torch.Tensor, logits: torch.Tensor):
    """
    Save x (input), y (gt), pred (binary) as NIfTI for visualization.
    x,y,logits shapes: (1,1,40,40,40)
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    x_np = x[0, 0].detach().cpu().numpy().astype(np.float32)
    y_np = y[0, 0].detach().cpu().numpy().astype(np.float32)
    pred_np = (torch.sigmoid(logits)[0, 0] > 0.5).detach().cpu().numpy().astype(np.float32)

    affine = np.eye(4, dtype=np.float32)
    nib.save(nib.Nifti1Image(x_np, affine), str(out_dir / f"{name}_image.nii.gz"))
    nib.save(nib.Nifti1Image(y_np, affine), str(out_dir / f"{name}_gt.nii.gz"))
    nib.save(nib.Nifti1Image(pred_np, affine), str(out_dir / f"{name}_pred.nii.gz"))


# -------------------------
# Gradient / parameter norm utilities
# -------------------------
@torch.no_grad()
def compute_grad_norm(model: nn.Module) -> float:
    """L2 norm of all gradients (only params with grad)."""
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.data.norm(2).item() ** 2
    return total ** 0.5


@torch.no_grad()
def compute_param_norm(model: nn.Module) -> float:
    """L2 norm of all parameters."""
    total = 0.0
    for p in model.parameters():
        total += p.data.norm(2).item() ** 2
    return total ** 0.5


# -------------------------
# Encoder / init
# -------------------------
def build_encoder(cfg: dict):
    if cfg["model"]["encoder"] == "resnet18_3d_gn":
        return resnet18_3d_gn(emb_dim=int(cfg["model"]["emb_dim"]))
    raise ValueError(f"Unknown encoder: {cfg['model']['encoder']}")


def load_pretrained_encoder(encoder: nn.Module, ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if "encoder_state" not in ckpt:
        raise RuntimeError(f"Checkpoint missing encoder_state: {ckpt_path}")
    encoder.load_state_dict(ckpt["encoder_state"], strict=True)
    return ckpt


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -------------------------
# CV folds
# -------------------------
def load_folds(folds_json: str) -> dict:
    with open(folds_json, "r") as f:
        return json.load(f)


def make_train_val_from_train_files(train_files: list[str], val_ratio: float, seed: int):
    rng = np.random.RandomState(int(seed))
    train_files = list(train_files)
    rng.shuffle(train_files)
    n = len(train_files)
    n_val = int(round(val_ratio * n))
    val_files = train_files[:n_val]
    tr_files = train_files[n_val:]
    return tr_files, val_files


# -------------------------
# Label fraction (TRAIN only)
# -------------------------
def subsample_train_files(train_files: list[str], frac: float, seed: int) -> list[str]:
    """
    Deterministically subsample TRAIN files to a given fraction.
    Keeps val/test fixed (you call this AFTER making train/val).
    """
    frac = float(frac)
    if frac >= 0.999:
        return list(train_files)

    rng = np.random.RandomState(int(seed))
    train_files = list(train_files)
    rng.shuffle(train_files)

    n = len(train_files)
    m = max(1, int(round(frac * n)))
    return train_files[:m]


# -------------------------
# Main
# -------------------------
def main(cfg_path: str):
    cfg = yaml.safe_load(open(cfg_path, "r"))
    seed = int(cfg.get("seed", 0))
    set_seed(seed)

    out_dir = Path(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(cfg.get("device", "cuda"))
    use_wandb = cfg.get("wandb", {}).get("enable", False)

    # ---- CV config (required) ----
    if "cv" not in cfg:
        raise RuntimeError("Missing cfg.cv. For CV runs, include cv: {folds_json, fold, val_ratio}.")

    folds_obj = load_folds(cfg["cv"]["folds_json"])
    fold_id = int(cfg["cv"]["fold"])
    val_ratio = float(cfg["cv"].get("val_ratio", 0.15))

    fold = folds_obj["folds"][fold_id]
    train_files_full = fold["train"]
    test_files = fold["test"]

    # validation derived ONLY from training fold
    train_files, val_files = make_train_val_from_train_files(
        train_files_full, val_ratio=val_ratio, seed=seed + fold_id
    )

    # ---- Label fraction (subsample TRAIN only; keep val/test fixed) ----
    label_frac_cfg = cfg.get("label_frac", {})
    use_label_frac = bool(label_frac_cfg.get("enable", False))
    label_frac = float(label_frac_cfg.get("frac", 1.0))

    frac_seed = None
    if use_label_frac:
        frac_seed = int(seed) + 1000 * int(fold_id) + int(round(label_frac * 100))
        before_n = len(train_files)
        train_files = subsample_train_files(train_files, label_frac, frac_seed)
        print(f"[LABEL_FRAC] frac={label_frac:.2f} {before_n} -> {len(train_files)} train samples (seed={frac_seed})")

    print(f"[CV] fold={fold_id} train={len(train_files)} val={len(val_files)} test={len(test_files)}")

    # ---- W&B init ----
    # if use_wandb:
    #     wandb.init(
    #         project=cfg["wandb"].get("project", "SPECT-MPI-Finetune-CV"),
    #         name=cfg["wandb"].get("run_name", None),
    #         config=cfg,
    #         dir=str(out_dir),
    #     )
    #     wandb.config.update(
    #         {
    #             "cv/fold": fold_id,
    #             "cv/val_ratio": val_ratio,
    #             "cv/n_total": int(folds_obj.get("n", -1)),
    #             "label_frac/enable": use_label_frac,
    #             "label_frac/frac": label_frac,
    #             "label_frac/seed": frac_seed,
    #             "label_frac/n_train": len(train_files),
    #         },
    #         allow_val_change=True,
    #     )
    #     wandb.define_metric("val/dice", summary="max")
    #     wandb.define_metric("val/iou", summary="max")
    #     wandb.define_metric("val/boundary_f1", summary="max")
    #     wandb.define_metric("val/precision", summary="max")
    #     wandb.define_metric("val/recall", summary="max")
    #     wandb.define_metric("epoch/train_loss", summary="min")

    # ---- Data ----
    images_dir = cfg["data"]["images_dir"]
    labels_dir = cfg["data"]["labels_dir"]

    ds_tr = PairSegDataset(images_dir, labels_dir, cfg, train_files)
    ds_va = PairSegDataset(images_dir, labels_dir, cfg, val_files)
    ds_te = PairSegDataset(images_dir, labels_dir, cfg, test_files)

    dl_tr = DataLoader(
        ds_tr,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["data"]["num_workers"]),
        pin_memory=True,
        drop_last=False,
    )
    dl_va = DataLoader(
        ds_va,
        batch_size=1,
        shuffle=False,
        num_workers=max(1, int(cfg["data"]["num_workers"]) // 2),
        pin_memory=True,
        drop_last=False,
    )
    dl_te = DataLoader(
        ds_te,
        batch_size=1,
        shuffle=False,
        num_workers=max(1, int(cfg["data"]["num_workers"]) // 2),
        pin_memory=True,
        drop_last=False,
    )

    # ---- Model ----
    encoder = build_encoder(cfg).to(device)
    init_mode = cfg["finetune"].get("init", "scratch")

    if init_mode == "mae":
        load_pretrained_encoder(encoder, cfg["finetune"]["pretrained_ckpt"])
        print("[INIT] Encoder initialized from MAE checkpoint")
    elif init_mode == "scratch":
        print("[INIT] Encoder initialized from scratch (random init)")
    else:
        raise ValueError(f"Unknown finetune.init: {init_mode}")

    head = SegHeadFromResNetFeatures(base=int(cfg["finetune"].get("head_base", 128))).to(device)

    # ---- Loss ----
    bce = nn.BCEWithLogitsLoss()
    dice_loss = SoftDiceLoss()
    dice_w = float(cfg["finetune"].get("dice_weight", 0.5))

    # ---- Train config ----
    epochs = int(cfg["train"]["epochs"])
    freeze_epochs = int(cfg["finetune"].get("freeze_epochs", 0))
    lr_head = float(cfg["train"]["lr_head"])
    lr_enc = float(cfg["train"]["lr_encoder"])
    wd = float(cfg["train"]["weight_decay"])
    amp = bool(cfg["train"].get("amp", True))

    log_every_steps = int(cfg["train"].get("log_every", 0))
    save_examples_every = int(cfg["finetune"].get("save_examples_every", 50))
    num_examples = int(cfg["finetune"].get("num_examples", 3))
    dead_neuron_every = int(cfg["train"].get("dead_neuron_every", 20))

    # scratch sanity: do not freeze a random encoder
    if init_mode == "scratch" and freeze_epochs > 0:
        print("[WARN] scratch init: forcing freeze_epochs=0 (freezing random encoder is invalid baseline).")
        freeze_epochs = 0

    scaler = GradScaler(enabled=amp)
    global_step = 0
    best_val_dice = -1.0
    best_path = out_dir / "best.pt"

    # ----------------------------------------------------------------
    # FIX: Build optimizer ONCE before the training loop.
    #
    # Group 0 = head   (always trains at lr_head)
    # Group 1 = encoder (lr=0 while frozen, lr=lr_enc once unfrozen)
    #
    # This preserves Adam's first-moment (m) and second-moment (v)
    # estimates across epochs, which is critical for stable convergence.
    # Rebuilding every epoch resets m and v to zero, causing oversized
    # updates at the start of each epoch (v ≈ 0 → large effective LR).
    # ----------------------------------------------------------------
    initial_enc_lr = 0.0 if freeze_epochs > 0 else lr_enc
    opt = torch.optim.AdamW([
        {"params": head.parameters(),    "lr": lr_head,       "weight_decay": wd},
        {"params": encoder.parameters(), "lr": initial_enc_lr, "weight_decay": wd},
    ])

    def run_eval(dl, split_name: str, epoch: int, save_examples: bool):
        encoder.eval()
        head.eval()

        ms = {"dice": [], "iou": [], "precision": [], "recall": [], "vol_ratio": []}
        bf1_all = []
        saved = 0

        with torch.no_grad():
            for x, y, fname in dl:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                feat = encoder.forward_features(x)
                logits = head(feat)

                m = metrics_binary(logits, y)
                for k in ms:
                    ms[k].extend(m[k])

                bf1_samples = boundary_f1_3d(logits, y)
                bf1_all.extend(bf1_samples)

                if save_examples and saved < num_examples:
                    ex_dir = out_dir / "examples" / split_name / f"epoch_{epoch:03d}"
                    save_pred_example(ex_dir, Path(fname[0]).stem, x, y, logits)
                    saved += 1

        result = {}
        for k in ms:
            arr = np.array(ms[k])
            result[f"{split_name}/{k}"] = float(arr.mean())
            result[f"{split_name}/{k}_std"] = float(arr.std())

        bf1_arr = np.array(bf1_all)
        result[f"{split_name}/boundary_f1"] = float(bf1_arr.mean())
        result[f"{split_name}/boundary_f1_std"] = float(bf1_arr.std())

        return result

    # -------------------------
    # Training loop
    # -------------------------
    for epoch in range(1, epochs + 1):
        encoder.train()
        head.train()

        frozen = (epoch <= freeze_epochs)
        for p in encoder.parameters():
            p.requires_grad = not frozen

        # ---- Adjust encoder LR; optimizer object stays the same ----
        opt.param_groups[1]["lr"] = 0.0 if frozen else lr_enc

        epoch_loss_sum = 0.0
        epoch_bce_sum = 0.0
        epoch_dice_sum = 0.0
        epoch_steps = 0
        epoch_samples = 0
        epoch_grad_norms_enc = []
        epoch_grad_norms_head = []
        epoch_start_time = time.time()

        for x, y, _fname in dl_tr:
            batch_size_actual = x.shape[0]
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with autocast(enabled=amp):
                feat = encoder.forward_features(x)
                logits = head(feat)
                loss_bce = bce(logits, y)
                loss_dice = dice_loss(logits, y)
                loss = (1.0 - dice_w) * loss_bce + dice_w * loss_dice

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()

            scaler.unscale_(opt)
            enc_grad_norm = compute_grad_norm(encoder)
            head_grad_norm = compute_grad_norm(head)
            epoch_grad_norms_enc.append(enc_grad_norm)
            epoch_grad_norms_head.append(head_grad_norm)

            scaler.step(opt)
            scaler.update()

            lv = float(loss.item())
            lv_bce = float(loss_bce.item())
            lv_dice = float(loss_dice.item())
            epoch_loss_sum += lv * batch_size_actual
            epoch_bce_sum += lv_bce * batch_size_actual
            epoch_dice_sum += lv_dice * batch_size_actual
            epoch_steps += 1
            epoch_samples += batch_size_actual

            if use_wandb and log_every_steps and (global_step % log_every_steps == 0):
                wandb.log(
                    {
                        "train/loss_step": lv,
                        "train/bce_step": lv_bce,
                        "train/dice_loss_step": lv_dice,
                        "train/grad_norm_encoder": enc_grad_norm,
                        "train/grad_norm_head": head_grad_norm,
                        "train/encoder_frozen": int(frozen),
                        "epoch": epoch,
                    },
                    step=global_step,
                )

            global_step += 1

        epoch_elapsed = time.time() - epoch_start_time
        epoch_train_loss = epoch_loss_sum / max(1, epoch_samples)
        epoch_train_bce = epoch_bce_sum / max(1, epoch_samples)
        epoch_train_dice = epoch_dice_sum / max(1, epoch_samples)
        throughput = epoch_samples / max(epoch_elapsed, 1e-6)

        print(f"[E{epoch:03d}] train_loss={epoch_train_loss:.6f} "
              f"bce={epoch_train_bce:.6f} dice_l={epoch_train_dice:.6f} "
              f"frozen={int(frozen)} throughput={throughput:.1f} samples/s")

        save_ex = (epoch == 1) or (epoch % save_examples_every == 0) or (epoch == epochs)
        val_metrics = run_eval(dl_va, "val", epoch, save_examples=save_ex)

        val_dice = val_metrics["val/dice"]
        print(f"          val_dice={val_dice:.4f} val_iou={val_metrics['val/iou']:.4f} "
              f"val_bf1={val_metrics['val/boundary_f1']:.4f}")

        enc_param_norm = compute_param_norm(encoder)
        head_param_norm = compute_param_norm(head)

        dead_neuron_metrics = {}
        if epoch == 1 or epoch % dead_neuron_every == 0 or epoch == epochs:
            dn = compute_dead_neuron_ratio(encoder, head, dl_va, device, max_batches=10)
            dead_neuron_metrics["dead_neuron/global_ratio"] = dn["dead_neuron/global_ratio"]
            if use_wandb and dn["dead_neuron/per_layer"]:
                dn_table = wandb.Table(columns=["layer", "dead_ratio"])
                for layer_name, ratio in dn["dead_neuron/per_layer"].items():
                    dn_table.add_data(layer_name, ratio)
                dead_neuron_metrics["dead_neuron/per_layer_table"] = dn_table

        actual_lrs = {f"optim/lr_group_{i}": pg["lr"] for i, pg in enumerate(opt.param_groups)}

        # if use_wandb:
        #     log_dict = {
        #         "epoch": epoch,
        #         "epoch/train_loss": epoch_train_loss,
        #         "epoch/train_bce": epoch_train_bce,
        #         "epoch/train_dice_loss": epoch_train_dice,
        #         "epoch/grad_norm_encoder_mean": float(np.mean(epoch_grad_norms_enc)) if epoch_grad_norms_enc else 0.0,
        #         "epoch/grad_norm_encoder_max": float(np.max(epoch_grad_norms_enc)) if epoch_grad_norms_enc else 0.0,
        #         "epoch/grad_norm_head_mean": float(np.mean(epoch_grad_norms_head)) if epoch_grad_norms_head else 0.0,
        #         "epoch/grad_norm_head_max": float(np.max(epoch_grad_norms_head)) if epoch_grad_norms_head else 0.0,
        #         "epoch/param_norm_encoder": enc_param_norm,
        #         "epoch/param_norm_head": head_param_norm,
        #         "epoch/throughput_samples_per_sec": throughput,
        #         "epoch/time_sec": epoch_elapsed,
        #         "epoch/encoder_frozen": int(frozen),
        #         "epoch/train_val_loss_ratio": epoch_train_loss / max(1.0 - val_dice, 1e-8),
        #     }
        #     log_dict.update(val_metrics)
        #     log_dict.update(actual_lrs)
        #     log_dict.update(dead_neuron_metrics)
        #     wandb.log(log_dict, step=global_step)

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(
                {
                    "epoch": epoch,
                    "cfg": cfg,
                    "init_mode": init_mode,
                    "pretrained_from": (cfg["finetune"]["pretrained_ckpt"] if init_mode == "mae" else None),
                    "encoder_name": cfg["model"]["encoder"],
                    "encoder_state": encoder.state_dict(),
                    "head_state": head.state_dict(),
                    "best_val_dice": best_val_dice,
                    "cv": {"fold": fold_id, "val_ratio": val_ratio},
                    "label_frac": {"enable": use_label_frac, "frac": label_frac, "seed": frac_seed},
                    "files_split": {"train": train_files, "val": val_files, "test": test_files},
                    "global_step": global_step,
                },
                best_path,
            )
            print(f"Saved best: {best_path} (val_dice={best_val_dice:.4f})")

    # -------------------------
    # TEST: evaluate BEST checkpoint
    # -------------------------
    print("\n=== Loading BEST checkpoint for TEST evaluation ===")
    best_ckpt = torch.load(best_path, map_location="cpu")
    encoder.load_state_dict(best_ckpt["encoder_state"], strict=True)
    head.load_state_dict(best_ckpt["head_state"], strict=True)

    print("=== Final TEST evaluation (unseen fold test, best-by-val) ===")
    test_metrics = run_eval(dl_te, "test", epoch=int(best_ckpt["epoch"]), save_examples=True)
    for k, v in test_metrics.items():
        print(f"{k}: {v:.4f}")

    best_ckpt["test_metrics"] = test_metrics
    torch.save(best_ckpt, best_path)

    # if use_wandb:
    #     wandb.log(test_metrics, step=global_step)
    #     wandb.run.summary["best_val_dice"] = best_val_dice
    #     wandb.run.summary["best_val_epoch"] = int(best_ckpt["epoch"])
    #     for k, v in test_metrics.items():
    #         wandb.run.summary[k] = v
    #     wandb.finish()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=str, required=True)
    args = ap.parse_args()
    main(args.cfg)