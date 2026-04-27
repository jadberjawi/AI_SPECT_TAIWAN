import os
import glob
import argparse
import numpy as np
import SimpleITK as sitk

from core import find_cardiac_center, crop_with_center, normalize_and_suppress_edges


def process_all(volume_dir, mask_dir, output_dir, crop_size_mm):
    """Process all labeled volume+mask pairs."""
    vol_out = os.path.join(output_dir, "volumes")
    mask_out = os.path.join(output_dir, "masks")
    os.makedirs(vol_out, exist_ok=True)
    os.makedirs(mask_out, exist_ok=True)

    volume_files = sorted(glob.glob(os.path.join(volume_dir, "*.nii.gz")))

    print(f"Processing {len(volume_files)} labeled samples")
    print("=" * 60)

    for vol_path in volume_files:
        filename = os.path.basename(vol_path)
        mask_path = os.path.join(mask_dir, filename)

        if not os.path.exists(mask_path):
            print(f"[SKIP] {filename}: No mask")
            continue

        print(f"{filename}")

        volume_sitk = sitk.ReadImage(vol_path, sitk.sitkFloat32)
        mask_sitk = sitk.ReadImage(mask_path, sitk.sitkUInt8)

        # Find center from VOLUME only
        center = find_cardiac_center(volume_sitk)

        # Crop both with same center
        cropped_vol = crop_with_center(volume_sitk, center, crop_size_mm)
        cropped_mask = crop_with_center(mask_sitk, center, crop_size_mm)

        # Normalize volume only
        normalized_vol = normalize_and_suppress_edges(cropped_vol)

        # Save volume
        sitk.WriteImage(normalized_vol, os.path.join(vol_out, filename))

        # Save mask (ensure uint8)
        mask_array = sitk.GetArrayFromImage(cropped_mask).astype(np.uint8)
        final_mask = sitk.GetImageFromArray(mask_array)
        final_mask.CopyInformation(cropped_mask)
        sitk.WriteImage(final_mask, os.path.join(mask_out, filename))

    print("\nDone!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--volume_dir", required=True)
    parser.add_argument("--mask_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--crop_size", type=float, nargs=3, default=[192.0, 192.0, 192.0])
    args = parser.parse_args()

    process_all(args.volume_dir, args.mask_dir, args.output_dir, tuple(args.crop_size))


if __name__ == "__main__":
    main()