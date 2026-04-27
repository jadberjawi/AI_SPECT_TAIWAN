# preprocess.py

import os
import glob
import argparse
import numpy as np
import SimpleITK as sitk


def normalize_volume(sitk_image):
    """99th percentile intensity normalization to [0, 1]."""
    volume = sitk.GetArrayFromImage(sitk_image)
    
    nonzero = volume[volume > 0]
    if nonzero.size == 0:
        p99 = np.max(volume) if np.max(volume) > 0 else 1.0
    else:
        p99 = np.percentile(nonzero, 99.0)
    
    normalized = np.clip(volume, 0.0, p99) / p99
    
    out_sitk = sitk.GetImageFromArray(normalized.astype(np.float32))
    out_sitk.CopyInformation(sitk_image)
    
    return out_sitk


def process(input_dir, output_dir):
    """Normalize all volumes in directory."""
    os.makedirs(output_dir, exist_ok=True)
    
    volume_files = sorted(glob.glob(os.path.join(input_dir, "*.nii.gz")))
    print(f"Processing {len(volume_files)} volumes")
    
    for vol_path in volume_files:
        filename = os.path.basename(vol_path)
        
        volume_sitk = sitk.ReadImage(vol_path, sitk.sitkFloat32)
        normalized = normalize_volume(volume_sitk)
        sitk.WriteImage(normalized, os.path.join(output_dir, filename))
        
        print(f"  {filename}")
    
    print("Done!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    
    process(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()