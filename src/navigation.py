import os
import re
import numpy as np
import warnings
from skimage.filters import sobel_h, sobel_v
from skimage.morphology import opening, closing, disk, remove_small_objects
from skimage.measure import label, regionprops
from src.parser import read_scan_file

def stitch_survey_scans(folder_path: str, prefix: str = "Task_2."):
    """
    Finds all survey scans in a folder, extracts their grid coordinates from the filename,
    aligns their Z-backgrounds, and stitches them into a single macroscopic scan.
    """
    files = [f for f in os.listdir(folder_path) if f.startswith(prefix) and f.lower().endswith((".safm", ".npy"))]
    
    if not files:
        raise FileNotFoundError(f"No survey files found in {folder_path} with prefix {prefix}")

    tile_info = []
    for f in files:
        match = re.search(r"\.\d+_(\d+)\.(\d+)_", f)
        if match:
            r_idx, c_idx = int(match.group(1)), int(match.group(2))
            tile_info.append((r_idx, c_idx, f))
            
    if not tile_info:
        raise ValueError("Could not parse grid coordinates from filenames.")

    # Load and Z-align all tiles using median subtraction
    tiles = {}
    for r, c, filename in tile_info:
        meta, data = read_scan_file(os.path.join(folder_path, filename), channel=0)
        z_data = data["data"].astype(float)
        z_data = z_data - np.median(z_data)
        tiles[(r, c)] = z_data

    all_rows = sorted(list(set(r for r, c in tiles.keys())))
    all_cols = sorted(list(set(c for r, c in tiles.keys())))
    
    sample_tile = next(iter(tiles.values()))
    tile_shape = sample_tile.shape
    blank_tile = np.zeros(tile_shape)

    stitched_rows = []
    for r in all_rows:
        row_tiles = []
        for c in all_cols:
            row_tiles.append(tiles.get((r, c), blank_tile))
        stitched_rows.append(np.hstack(row_tiles))
        
    stitched_image = np.vstack(stitched_rows)
    return stitched_image

def locate_marker_centroid(stitched_scan: np.ndarray):
    """
    Uses Sobel gradient filtering to ignore horizontal stripe artifacts and 
    locates the geometric center of the navigation marker on the stitched scan.
    """
    # Suppress horizontal stripe noise
    edge = np.clip(np.abs(sobel_v(stitched_scan)) - 0.7 * np.abs(sobel_h(stitched_scan)), 0, None)
    
    # 90th percentile threshold
    thresh_val = np.percentile(edge, 90)
    binary_mask = edge > thresh_val
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        binary_mask = opening(binary_mask, disk(2))
        binary_mask = closing(binary_mask, disk(3))
        binary_mask = remove_small_objects(binary_mask, min_size=30)
        
    labeled_mask = label(binary_mask)
    regions = regionprops(labeled_mask)
    
    img_h, img_w = stitched_scan.shape
    center_y, center_x = img_h / 2.0, img_w / 2.0

    # Notebook's exact filter: reject very wide horizontal noise bands
    candidates = []
    for r in regions:
        h = r.bbox[2] - r.bbox[0]
        w = r.bbox[3] - r.bbox[1]
        aspect_ratio_w_over_h = w / max(h, 1)
        
        if 30 < r.area < 5000 and aspect_ratio_w_over_h < 3.0:
            candidates.append(r)
            
    # Robust fallback if candidate filtering is overly strict
    if not candidates:
        candidates = [r for r in regions if r.area > 20]
        
    if not candidates:
        # Ultimate fallback to image center if no edges survive
        return center_x, center_y, edge
        
    # Find candidate closest to image center
    best_candidate = min(candidates, key=lambda r: (r.centroid[0] - center_y)**2 + (r.centroid[1] - center_x)**2)
    
    y0, x0 = best_candidate.centroid
    return x0, y0, edge