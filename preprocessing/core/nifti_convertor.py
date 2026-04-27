# convert_dicom_to_nifti.py

import os
import argparse
import SimpleITK as sitk


def convert_series(dicom_dir, output_dir):
    """Convert DICOM series to NIfTI without any cropping."""
    os.makedirs(output_dir, exist_ok=True)
    
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(dicom_dir)
    
    if not series_ids:
        print(f"No DICOM series found in {dicom_dir}")
        return
    
    for sid in series_ids:
        dicom_names = reader.GetGDCMSeriesFileNames(dicom_dir, sid)
        if not dicom_names:
            continue
        
        reader.SetFileNames(sorted(dicom_names))
        image = reader.Execute()
        
        # Get output filename from first DICOM
        base_name = os.path.splitext(os.path.basename(dicom_names[0]))[0]
        out_path = os.path.join(output_dir, f"{base_name}.nii.gz")
        
        sitk.WriteImage(image, out_path)
        print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dicom_dir", required=True, help="Directory containing DICOM files")
    parser.add_argument("--output_dir", required=True, help="Output directory for NIfTI files")
    args = parser.parse_args()
    
    convert_series(args.dicom_dir, args.output_dir)


if __name__ == "__main__":
    main()