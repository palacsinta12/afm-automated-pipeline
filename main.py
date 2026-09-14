import argparse
import os
import warnings

import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TF backend warnings

from src.parser import read_scan_file
from src.navigation import stitch_survey_scans, locate_marker_centroid
from src.correction import plane_correction, line_correction
from src.image_ops import extract_background_mask, align_image, extract_digit_crops
from src.inference import preprocess_digit_for_mnist, load_and_predict


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Automated AFM Data Processing Pipeline")
    parser.add_argument("--survey-dir", type=str, required=True, help="Dir containing 3x3 survey scans")
    parser.add_argument("--highres", type=str, required=True, help="Path to high-res scan")
    parser.add_argument("--model", type=str, required=True, help="Path to trained CNN model")
    parser.add_argument("--save-plots", type=str, default=None, help="Save debug images to this dir")
    return parser


def main(survey_dir: str, highres_file: str, model_file: str, save_plots_dir: str = None):
    print("\n--- PHASE 1: MACROSCOPIC NAVIGATION ---")
    print(f"[*] Stitching survey scans from: {survey_dir}")
    stitched_img = stitch_survey_scans(survey_dir, prefix="Task_2.")
    
    print("[*] Detecting marker centroid via Sobel gradients...")
    cx, cy, edge_map = locate_marker_centroid(stitched_img)
    print(f"    -> Marker Centroid found at pixel (X: {cx:.1f}, Y: {cy:.1f})")
    
    print("\n--- PHASE 2: NANOSCALE DATA PROCESSING ---")
    print(f"[*] Loading high-resolution scan: {highres_file}")
    meta, data_hr = read_scan_file(highres_file, channel=0)
    raw_scan = data_hr["data"].astype(float)
    x = np.linspace(0, float(data_hr.get("xreal", 1)), raw_scan.shape[1])
    y = np.linspace(0, float(data_hr.get("yreal", 1)), raw_scan.shape[0])

    print("[*] Performing Masked Artifact Correction (Plane & Line)...")
    bg_mask = extract_background_mask(raw_scan)
    scan_planed, _ = plane_correction(raw_scan, x, y, mask=bg_mask)
    scan_flattened = line_correction(scan_planed, x, p=2, axis=0, mask=bg_mask)

    print("[*] Morphological Segmentation & Geometric Alignment...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from skimage.filters import gaussian, threshold_otsu
        from skimage.morphology import opening, closing, disk, remove_small_objects, remove_small_holes
        scan_smooth = gaussian(scan_flattened, sigma=1)
        binary_mask = scan_smooth > threshold_otsu(scan_smooth)
        binary_mask = remove_small_holes(remove_small_objects(closing(opening(binary_mask, disk(2)), disk(2)), 50), 30)

    scan_aligned, mask_aligned = align_image(scan_smooth, binary_mask)
    digit_crops = extract_digit_crops(scan_aligned, mask_aligned)
    print(f"    -> Found {len(digit_crops)} structural digits.")

    print("\n--- PHASE 3: DEEP LEARNING INFERENCE ---")
    if len(digit_crops) == 4: # Or len(digit_crops) > 0
        print("[*] Preprocessing digits (Center-of-Mass alignment)...")
        preprocessed_digits = [preprocess_digit_for_mnist(digit) for digit in digit_crops]
        
        labels, probabilities = load_and_predict(model_file, preprocessed_digits)
        marker_string = "".join(str(lbl) for lbl in labels)
        
        print("==================================================")
        print(f"🎯 PREDICTED AFM MARKER STRING: {marker_string}")
        
        # Configurable Coordinate Translation Logic
        grid_spacing_um = 200 # Configurable physical spacing
        coord_x = int(marker_string[:2]) * grid_spacing_um
        coord_y = int(marker_string[2:]) * grid_spacing_um
        print(f"📍 CHIP COORDINATES DEDUCED: X = {coord_x} µm, Y = {coord_y} µm")
        print("==================================================\n")
    else:
        print(f"[!] Warning: Expected 4 digits, but found {len(digit_crops)}. Skipping inference.")
        preprocessed_digits, labels, probabilities = [], [], None

    if save_plots_dir:
        os.makedirs(save_plots_dir, exist_ok=True)
        from src.visualize import plot_scan_correction, plot_extraction, plot_cnn_results, plot_marker_detection
        plot_marker_detection(stitched_img, cx, cy, edge_map, save_plots_dir)
        plot_scan_correction(raw_scan, bg_mask, scan_flattened, save_plots_dir)
        plot_extraction(scan_aligned, mask_aligned, save_plots_dir)
        if len(preprocessed_digits) > 0:
            plot_cnn_results(preprocessed_digits, labels, probabilities, save_plots_dir)


def run_cli(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    main(args.survey_dir, args.highres, args.model, args.save_plots)


if __name__ == "__main__":
    run_cli()