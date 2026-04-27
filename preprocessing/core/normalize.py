import numpy as np
import SimpleITK as sitk


def normalize_and_suppress_edges(sitk_image, safe_radius_mm=55.0, fade_mm=20.0):
    """
    Apply spherical edge suppression and intensity normalization.
    
    Args:
        sitk_image: SimpleITK image (should be cropped first)
        safe_radius_mm: Radius of fully preserved region
        fade_mm: Width of fade zone
        
    Returns:
        normalized_sitk: Normalized SimpleITK image [0, 1]
    """
    volume = sitk.GetArrayFromImage(sitk_image)
    spacing = sitk_image.GetSpacing()
    sx, sy, sz = spacing
    D, H, W = volume.shape

    # Distance from center in mm
    z, y, x = np.ogrid[:D, :H, :W]
    center_z, center_y, center_x = D/2, H/2, W/2

    dist_sq = (
        ((z - center_z) * sz) ** 2
        + ((y - center_y) * sy) ** 2
        + ((x - center_x) * sx) ** 2
    )
    distance_mm = np.sqrt(dist_sq)

    # Build spherical mask
    weight_mask = np.ones_like(volume, dtype=np.float32)

    # Fade zone: cosine falloff
    fade_zone = (distance_mm > safe_radius_mm) & (distance_mm <= (safe_radius_mm + fade_mm))
    normalized_dist = (distance_mm[fade_zone] - safe_radius_mm) / fade_mm
    weight_mask[fade_zone] = 0.5 * (1 + np.cos(normalized_dist * np.pi))

    # Beyond fade: zero
    weight_mask[distance_mm > (safe_radius_mm + fade_mm)] = 0.0

    # Apply mask
    suppressed_vol = volume * weight_mask

    # Normalize to [0, 1]
    nonzero = suppressed_vol[suppressed_vol > 0]
    if nonzero.size == 0:
        p99 = np.max(suppressed_vol) if np.max(suppressed_vol) > 0 else 1.0
    else:
        p99 = np.percentile(nonzero, 99.0)

    normalized_vol = np.clip(suppressed_vol, 0.0, p99) / p99

    # Rebuild with metadata
    normalized_sitk = sitk.GetImageFromArray(normalized_vol.astype(np.float32))
    normalized_sitk.CopyInformation(sitk_image)

    return normalized_sitk