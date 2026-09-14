import numpy as np
import warnings
from skimage.filters import gaussian, threshold_otsu
from skimage.morphology import opening, closing, disk, remove_small_objects, remove_small_holes
from skimage.measure import label, regionprops
from skimage.transform import rotate

def extract_background_mask(scan: np.ndarray, sigma: float = 1.0):
    """
    Uses image gradients to identify the flat background and isolate the
    structural foreground for later correction and segmentation stages.
    """
    z_smooth = gaussian(scan, sigma=sigma)
    dy, dx = np.gradient(z_smooth)
    grad_abs = np.sqrt(dx**2 + dy**2)

    grad_thresh = threshold_otsu(grad_abs)
    background_mask = grad_abs < grad_thresh

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        background_mask = opening(background_mask, disk(2))
        background_mask = closing(background_mask, disk(2))
        background_mask = remove_small_objects(background_mask, min_size=100)
        background_mask = remove_small_holes(background_mask, area_threshold=100)

    return background_mask

def align_image(scan: np.ndarray, binary_mask: np.ndarray):
    """
    Finds the dominant orientation marker, rotates the scan to align it,
    and mirrors the image to normalize the geometry before digit crops
    are extracted.
    """
    labeled_mask = label(binary_mask)
    regions = regionprops(labeled_mask)

    if not regions:
        raise ValueError("No connected regions found in mask.")

    largest_region = max(regions, key=lambda r: r.area)

    marker_orientation_rad = largest_region.orientation
    marker_orientation_deg = 180 - np.degrees(marker_orientation_rad)

    z_rot = rotate(scan, marker_orientation_deg, resize=True, preserve_range=True)
    mask_rot = rotate(binary_mask.astype(float), marker_orientation_deg, resize=True, preserve_range=True) > 0.5

    z_rot = np.fliplr(z_rot)
    mask_rot = np.fliplr(mask_rot)

    return z_rot, mask_rot

def extract_digit_crops(aligned_scan: np.ndarray, aligned_mask: np.ndarray, size_factor: float = 2.0):
    """
    Filters connected regions by size and returns the digit crops in a
    stable reading order suitable for downstream classification.
    """
    labeled_image = label(aligned_mask)
    regions = regionprops(labeled_image)

    valid_regions = [r for r in regions if 200 <= r.area <= 5000]

    if not valid_regions:
        return []

    bbox_areas = np.array([(r.bbox[2] - r.bbox[0]) * (r.bbox[3] - r.bbox[1]) for r in valid_regions])
    sorted_areas = np.sort(bbox_areas)
    n_small = max(1, len(sorted_areas) // 2)
    reference_area = np.median(sorted_areas[:n_small])

    digit_crops = []

    for r in valid_regions:
        box_area = (r.bbox[2] - r.bbox[0]) * (r.bbox[3] - r.bbox[1])
        if box_area <= size_factor * reference_area:
            clean_digit_mask = r.image.astype(float)

            min_row, min_col, max_row, max_col = r.bbox
            center_row = 0.5 * (min_row + max_row)
            center_col = 0.5 * (min_col + max_col)
            digit_crops.append({"img": clean_digit_mask, "row": center_row, "col": center_col})

    if len(digit_crops) > 0:
        median_row = np.median([d["row"] for d in digit_crops])
        digit_crops = sorted(digit_crops, key=lambda d: (d["row"] < median_row, d["col"]))

    return [d["img"] for d in digit_crops]