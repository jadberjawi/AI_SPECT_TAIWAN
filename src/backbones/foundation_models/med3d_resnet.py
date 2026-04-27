# src/backbones/med3d_resnet.py
from __future__ import annotations
from typing import Dict, Tuple, Any, Optional

import torch
import torch.nn as nn


def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock3D(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride=stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes, stride=1)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.relu(out + identity)
        return out


class ResNet3D(nn.Module):
    """
    ResNet3D backbone that exposes:
      - stem, layer1..layer4
      - forward_features(x): (B, C, D', H', W')
    """

    def __init__(self, block, layers, in_ch=1, stem_stride=1, use_maxpool=False):
        super().__init__()
        self.inplanes = 64

        self.conv1 = nn.Conv3d(in_ch, 64, kernel_size=7, stride=stem_stride, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1) if use_maxpool else nn.Identity()

        # expose a "stem" attribute to match your finetune code
        self.stem = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool)

        self.layer1 = self._make_layer(block, 64,  layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # not used for your pipeline, but common in checkpoints
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(512 * block.expansion, 1000)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1x1(self.inplanes, planes * block.expansion, stride=stride),
                nn.BatchNorm3d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride=stride, downsample=downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x  # (B, 512, d, h, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # standard classification forward (not used in your CADClassifier)
        f = self.forward_features(x)
        z = self.avgpool(f).flatten(1)
        return self.fc(z)


def med3d_resnet18(in_ch=1, stem_stride=1, use_maxpool=False) -> ResNet3D:
    return ResNet3D(BasicBlock3D, [2, 2, 2, 2], in_ch=in_ch, stem_stride=stem_stride, use_maxpool=use_maxpool)


def _extract_state_dict(ckpt: Any) -> Dict[str, torch.Tensor]:
    if isinstance(ckpt, dict):
        for k in ["state_dict", "model_state", "model", "net", "encoder_state"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
        # sometimes the dict is already a state_dict
        if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            return ckpt
    raise RuntimeError("Unrecognized checkpoint format (expected dict with a state_dict-like payload).")


def load_med3d_state_dict(model: nn.Module, ckpt_path: str, strict: bool = False) -> Tuple[list[str], list[str]]:
    """
    Loads MedicalNet/Med3D pretrained weights into our ResNet3D.
    - strips prefixes: 'module.', 'backbone.', 'encoder.', 'model.'
    - ignores fc layer by default (common mismatch)
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = _extract_state_dict(ckpt)

    cleaned = {}
    for k, v in state.items():
        kk = k
        for pref in ["module.", "backbone.", "encoder.", "model."]:
            if kk.startswith(pref):
                kk = kk[len(pref):]
        # ignore classifier head weights from pretraining
        if kk.startswith("fc."):
            continue
        cleaned[kk] = v

    missing, unexpected = model.load_state_dict(cleaned, strict=strict)
    return missing, unexpected