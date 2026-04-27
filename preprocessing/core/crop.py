import numpy as np
import SimpleITK as sitk
from scipy import ndimage

DEFAULT_CROP_SIZE_MM = (192.0, 192.0, 192.0)


def find_cardiac_center(sitk_image):
    """
    Find cardiac center from SPECT volume using Gaussian-weighted intensity.
    
    Args:
        sitk_image: SimpleITK image
        
    Returns:
        center: numpy array (Z, Y, X) voxel coordinates
    """
    volume = sitk.GetArrayFromImage(sitk_image)
    D, H, W = volume.shape

    z, y, x = np.ogrid[:D, :H, :W]
    center_z, center_y, center_x = D/2, H/2, W/2
    sigma = min(D, H, W) / 3.0
    dist_sq = ((z - center_z)**2 + (y - center_y)**2 + (x - center_x)**2)
    gaussian_mask = np.exp(-dist_sq / (2 * sigma**2))

    nonzero = volume[volume > 0]
    p99 = np.percentile(nonzero, 99.0) if nonzero.size > 0 else np.percentile(volume, 99.0)

    vol_clipped = np.clip(volume, 0, p99)
    weighted_vol = vol_clipped * gaussian_mask

    threshold = np.percentile(weighted_vol, 99.0)
    binary_mask = weighted_vol > threshold

    try:
        coords = ndimage.center_of_mass(binary_mask)
        if coords is None or not np.all(np.isfinite(coords)):
            raise ValueError
        center = np.array(coords).astype(int)
    except Exception:
        center = np.array([D//2, H//2, W//2])

    return center


def crop_with_center(sitk_image, center, crop_size_mm):
    """
    Crop a volume using a specified center point.
    
    Args:
        sitk_image: SimpleITK image (volume or mask)
        center: (Z, Y, X) voxel coordinates
        crop_size_mm: (X, Y, Z) physical size in mm
        
    Returns:
        cropped_sitk: Cropped SimpleITK image with preserved metadata
    """
    spacing = np.asarray(sitk_image.GetSpacing(), dtype=float)
    direction = sitk_image.GetDirection()
    volume = sitk.GetArrayFromImage(sitk_image)
    D, H, W = volume.shape

    crop_size_mm = np.asarray(crop_size_mm, dtype=float)
    crop_size_voxels = np.maximum(1, np.round(crop_size_mm / spacing).astype(int))
    cx, cy, cz = crop_size_voxels

    center_z, center_y, center_x = center

    z_start = center_z - cz // 2
    y_start = center_y - cy // 2
    x_start = center_x - cx // 2
    z_end = z_start + cz
    y_end = y_start + cy
    x_end = x_start + cx

    # Padding if crop extends beyond volume
    pad_z_before = max(0, -z_start)
    pad_y_before = max(0, -y_start)
    pad_x_before = max(0, -x_start)
    pad_z_after = max(0, z_end - D)
    pad_y_after = max(0, y_end - H)
    pad_x_after = max(0, x_end - W)

    # Valid extraction region
    valid_z_start = max(0, z_start)
    valid_y_start = max(0, y_start)
    valid_x_start = max(0, x_start)
    valid_z_end = min(D, z_end)
    valid_y_end = min(H, y_end)
    valid_x_end = min(W, x_end)

    cropped_vol = volume[valid_z_start:valid_z_end,
                         valid_y_start:valid_y_end,
                         valid_x_start:valid_x_end]

    cropped_vol = np.pad(
        cropped_vol,
        ((pad_z_before, pad_z_after),
         (pad_y_before, pad_y_after),
         (pad_x_before, pad_x_after)),
        mode="constant",
        constant_values=0
    )

    cropped_sitk = sitk.GetImageFromArray(cropped_vol)
    cropped_sitk.SetSpacing(spacing)
    cropped_sitk.SetDirection(direction)
    new_origin = sitk_image.TransformIndexToPhysicalPoint(
        (int(x_start), int(y_start), int(z_start))
    )
    cropped_sitk.SetOrigin(new_origin)

    return cropped_sitk


def crop_volume(sitk_image, crop_size_mm=DEFAULT_CROP_SIZE_MM):
    """
    Convenience function: find center and crop in one call.
    
    Args:
        sitk_image: SimpleITK image
        crop_size_mm: (X, Y, Z) physical size in mm
        
    Returns:
        cropped_sitk: Cropped SimpleITK image
        center: The center used (Z, Y, X)
    """
    center = find_cardiac_center(sitk_image)
    cropped = crop_with_center(sitk_image, center, crop_size_mm)
    return cropped, center