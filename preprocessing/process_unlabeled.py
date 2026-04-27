import os
import glob
import argparse
import SimpleITK as sitk

from core import crop_volume, normalize_and_suppress_edges


def process_all(input_dir, output_dir, crop_size_mm):
    """Process all unlabeled volumes."""
    os.makedirs(output_dir, exist_ok=True)

    volume_files = sorted(glob.glob(os.path.join(input_dir, "*.nii.gz")))

    print(f"Processing {len(volume_files)} unlabeled samples")
    print("=" * 60)

    for vol_path in volume_files:
        filename = os.path.basename(vol_path)
        print(f"{filename}")

        volume_sitk = sitk.ReadImage(vol_path, sitk.sitkFloat32)

        # Crop (center found automatically)
        cropped, _ = crop_volume(volume_sitk, crop_size_mm)

        # Normalize
        normalized = normalize_and_suppress_edges(cropped)

        # Save
        sitk.WriteImage(normalized, os.path.join(output_dir, filename))

    print("\nDone!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--crop_size", type=float, nargs=3, default=[160.0, 160.0, 160.0])
    args = parser.parse_args()

    process_all(args.input_dir, args.output_dir, tuple(args.crop_size))


if __name__ == "__main__":
    main()