import os
import sys
# Fix import path - must be before importing from core
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
import glob
import argparse
import numpy as np
import SimpleITK as sitk

from core import find_cardiac_center


def validate_single_sample(volume_sitk, mask_sitk, crop_size_mm, margin_warning_mm=10.0):
    """Check if LV mask is safely inside the proposed crop region."""
    spacing = np.asarray(volume_sitk.GetSpacing(), dtype=float)
    mask_array = sitk.GetArrayFromImage(mask_sitk)
    D, H, W = mask_array.shape

    center = find_cardiac_center(volume_sitk)

    crop_size_mm = np.asarray(crop_size_mm, dtype=float)
    crop_size_voxels = np.maximum(1, np.round(crop_size_mm / spacing).astype(int))
    cx, cy, cz = crop_size_voxels

    center_z, center_y, center_x = center

    crop_bounds = {
        'z': (center_z - cz // 2, center_z + cz // 2),
        'y': (center_y - cy // 2, center_y + cy // 2),
        'x': (center_x - cx // 2, center_x + cx // 2),
    }

    mask_coords = np.where(mask_array > 0)

    if len(mask_coords[0]) == 0:
        return False, {'error': 'Empty mask'}

    mask_bounds = {
        'z': (mask_coords[0].min(), mask_coords[0].max()),
        'y': (mask_coords[1].min(), mask_coords[1].max()),
        'x': (mask_coords[2].min(), mask_coords[2].max()),
    }

    issues = []
    spacing_map = {'x': spacing[0], 'y': spacing[1], 'z': spacing[2]}
    vol_sizes = {'z': D, 'y': H, 'x': W}

    for axis in ['z', 'y', 'x']:
        margin_v = margin_warning_mm / spacing_map[axis]
        crop_min, crop_max = crop_bounds[axis]
        mask_min, mask_max = mask_bounds[axis]

        effective_crop_min = max(0, crop_min)
        effective_crop_max = min(vol_sizes[axis], crop_max)

        if mask_min < effective_crop_min:
            issues.append(f'{axis.upper()}: mask {effective_crop_min - mask_min} voxels OUTSIDE [CRITICAL]')
        elif mask_min < effective_crop_min + margin_v:
            issues.append(f'{axis.upper()}: mask {mask_min - effective_crop_min:.1f} voxels from edge [WARNING]')

        if mask_max > effective_crop_max:
            issues.append(f'{axis.upper()}: mask {mask_max - effective_crop_max} voxels OUTSIDE [CRITICAL]')
        elif mask_max > effective_crop_max - margin_v:
            issues.append(f'{axis.upper()}: mask {effective_crop_max - mask_max:.1f} voxels from edge [WARNING]')

    is_safe = not any('[CRITICAL]' in issue for issue in issues)

    return is_safe, {'center': center.tolist(), 'issues': issues}


def validate_all(volume_dir, mask_dir, crop_size_mm):
    """Validate all labeled samples."""
    volume_files = sorted(glob.glob(os.path.join(volume_dir, "*.nii.gz")))

    if not volume_files:
        print(f"No .nii.gz files found in {volume_dir}")
        return False

    critical_failures = []
    warnings = []

    print(f"Validating {len(volume_files)} samples | Crop: {crop_size_mm} mm")
    print("=" * 60)

    for vol_path in volume_files:
        filename = os.path.basename(vol_path)
        mask_path = os.path.join(mask_dir, filename)

        if not os.path.exists(mask_path):
            print(f"[SKIP] {filename}: No mask")
            continue

        volume_sitk = sitk.ReadImage(vol_path, sitk.sitkFloat32)
        mask_sitk = sitk.ReadImage(mask_path, sitk.sitkUInt8)

        is_safe, details = validate_single_sample(volume_sitk, mask_sitk, crop_size_mm)

        if 'error' in details:
            print(f"[ERROR] {filename}: {details['error']}")
            critical_failures.append(filename)
        elif details['issues']:
            print(f"\n{filename}:")
            for issue in details['issues']:
                print(f"  - {issue}")
            if not is_safe:
                critical_failures.append(filename)
            else:
                warnings.append(filename)

    print("\n" + "=" * 60)
    print(f"Passed: {len(volume_files) - len(critical_failures) - len(warnings)}")
    print(f"Warnings: {len(warnings)}")
    print(f"Critical: {len(critical_failures)}")

    if critical_failures:
        print("\n** DO NOT PROCEED — mask data will be lost **")
        return False

    print("\n[OK] Safe to proceed.")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--volume_dir", required=True)
    parser.add_argument("--mask_dir", required=True)
    parser.add_argument("--crop_size", type=float, nargs=3, default=[192.0, 192.0, 192.0])
    args = parser.parse_args()

    validate_all(args.volume_dir, args.mask_dir, tuple(args.crop_size))


if __name__ == "__main__":
    main()