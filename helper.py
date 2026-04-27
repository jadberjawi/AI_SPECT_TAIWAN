import os
import nibabel as nib
import numpy as np

def inspect_nifti_dataset(images_dir, masks_dir):
    # Get all NIfTI files from the images directory, sorted alphabetically
    valid_extensions = ('.nii', '.nii.gz')
    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith(valid_extensions)])
    
    if not image_files:
        print(f"No NIfTI files found in {images_dir}")
        return

    for img_filename in image_files:
        img_path = os.path.join(images_dir, img_filename)
        
        # This assumes your masks have the exact same filename as your images. 
        # If they have a suffix (like '_mask'), adjust the filename string here.
        lbl_path = os.path.join(masks_dir, img_filename) 
        
        if not os.path.exists(lbl_path):
            print(f"{img_filename} - Warning: Corresponding mask not found in {masks_dir}")
            continue
            
        try:
            # Load the NIfTI files
            img_nii = nib.load(img_path)
            lbl_nii = nib.load(lbl_path)
            
            # Load the data into numpy arrays
            img_data = img_nii.get_fdata()
            lbl_data = lbl_nii.get_fdata()
            
            # Extract shapes
            img_shape = img_data.shape
            lbl_shape = lbl_data.shape
            
            # Calculate min and max for the image range
            img_min = np.min(img_data)
            img_max = np.max(img_data)
            
            # Find unique values in the mask
            lbl_unique = np.unique(lbl_data)
            
            # Format the output string exactly as requested
            print(f"{img_filename} img shape={img_shape} lbl shape={lbl_shape} img range=({img_min:.2f},{img_max:.2f}) lbl unique={lbl_unique}")
            
        except Exception as e:
            print(f"Error processing {img_filename}: {e}")

# --- Set your directory paths here ---
IMAGES_FOLDER = 'data/volumes'
MASKS_FOLDER = 'data/labels'

if __name__ == "__main__":
    inspect_nifti_dataset(IMAGES_FOLDER, MASKS_FOLDER)