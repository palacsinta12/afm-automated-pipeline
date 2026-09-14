# --- START OF FILE scripts/generate_synthetic_afm.py ---

import os
import argparse
import numpy as np
from skimage.transform import resize, rotate
from skimage.filters import gaussian

def get_computer_digit(target_label: int) -> np.ndarray:
    """Returns a continuous, nanofabricated-style digit using bicubic interpolation."""
    digits = {
        0: [[0,1,1,1,0],[1,0,0,0,1],[1,0,0,1,1],[1,0,1,0,1],[1,1,0,0,1],[1,0,0,0,1],[0,1,1,1,0]],
        1: [[0,0,1,0,0],[0,1,1,0,0],[0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0],[0,1,1,1,0]],
        2: [[0,1,1,1,0],[1,0,0,0,1],[0,0,0,0,1],[0,0,1,1,0],[0,1,0,0,0],[1,0,0,0,0],[1,1,1,1,1]],
        3: [[0,1,1,1,0],[1,0,0,0,1],[0,0,0,0,1],[0,0,1,1,0],[0,0,0,0,1],[1,0,0,0,1],[0,1,1,1,0]],
        4: [[0,0,0,1,0],[0,0,1,1,0],[0,1,0,1,0],[1,0,0,1,0],[1,1,1,1,1],[0,0,0,1,0],[0,0,0,1,0]],
        5: [[1,1,1,1,1],[1,0,0,0,0],[1,1,1,1,0],[0,0,0,0,1],[0,0,0,0,1],[1,0,0,0,1],[0,1,1,1,0]],
        6: [[0,1,1,1,0],[1,0,0,0,0],[1,0,0,0,0],[1,1,1,1,0],[1,0,0,0,1],[1,0,0,0,1],[0,1,1,1,0]],
        7: [[1,1,1,1,1],[0,0,0,0,1],[0,0,0,1,0],[0,0,1,0,0],[0,1,0,0,0],[1,0,0,0,0],[1,0,0,0,0]],
        8: [[0,1,1,1,0],[1,0,0,0,1],[1,0,0,0,1],[0,1,1,1,0],[1,0,0,0,1],[1,0,0,0,1],[0,1,1,1,0]],
        9: [[0,1,1,1,0],[1,0,0,0,1],[1,0,0,0,1],[0,1,1,1,1],[0,0,0,0,1],[0,0,0,0,1],[0,1,1,1,0]]
    }
    
    arr = np.array(digits.get(target_label, digits[0]), dtype=float)
    
    # Pad so bicubic interpolation blooms smoothly instead of clipping edges
    arr = np.pad(arr, pad_width=1, mode='constant', constant_values=0)
    
    # AFM origin="lower" means row 0 is the bottom. Flip it upright.
    arr = np.flipud(arr)
    
    # Bicubic interpolation (order=3) simulates smooth etching
    digit_img = resize(arr, (50, 50), anti_aliasing=True, order=3)
    if digit_img.max() > 0:
        digit_img /= digit_img.max()
        
    return digit_img


def generate_synthetic_data(output_dir="data/raw/synthetic", marker_digits="1337"):
    os.makedirs(output_dir, exist_ok=True)
    
    marker_digits = str(marker_digits).zfill(4)[:4]
    parsed_digits = [int(d) for d in marker_digits]
    
    print(f"Generating Ultra-Realistic Synthetic AFM data for marker [{marker_digits}]...")

    # ==========================================
    # 1. BUILD THE PERFECTLY ALIGNED TARGET
    # ==========================================
    canvas_size = 256
    desired = np.zeros((canvas_size, canvas_size))
    center = canvas_size // 2
    thickness = 3  
    
    # Dominant Vertical Arm (Length 236) & Minor Horizontal Arm (Length 176)
    desired[10:-10, center - thickness : center + thickness] = 1.0
    desired[center - thickness : center + thickness, 40:-40] = 1.0

    # Read Order: Top-Left, Top-Right, Bottom-Left, Bottom-Right
    # (Since origin="lower", Row 170 is physically the TOP of the image)
    quadrants = [
        (170, 40),   # Row High, Col Low
        (170, 170),  # Row High, Col High
        (40, 40),    # Row Low,  Col Low
        (40, 170),   # Row Low,  Col High
    ]
    
    for digit_label, (row, col) in zip(parsed_digits, quadrants):
        digit_img = get_computer_digit(digit_label)
        desired[row : row + 50, col : col + 50] = np.maximum(
            desired[row : row + 50, col : col + 50], digit_img
        )

    # ==========================================
    # 2. APPLY INVERSE PIPELINE (Create Raw Scan)
    # ==========================================
    # Your `align_image` will measure a 15-degree tilt and rotate by 165, then fliplr.
    # The mathematical inverse to perfectly reconstruct `desired` is to fliplr, then rotate 195.
    raw_scan = rotate(np.fliplr(desired), 195, resize=False)

    # ==========================================
    # 3. ADD EXTREME PHYSICAL REALISM
    # ==========================================
    # Tip Convolution (Smooths and blooms the sharp structures)
    raw_scan = gaussian(raw_scan, sigma=2.0) * 150.0

    # PID Feedback Shadowing (Trailing edge drops slightly)
    _, dx = np.gradient(raw_scan)
    raw_scan += np.clip(dx, 0, None) * 0.6 

    # Substrate Tilt & Piezo Scanner Lines
    x = np.linspace(0, 1, canvas_size)
    X, Y = np.meshgrid(x, x)
    tilt = 400 * X + 150 * Y
    highres_lines = np.repeat(np.random.normal(0, 6.0, (canvas_size, 1)), canvas_size, axis=1)

    # Random Surface Debris (Blobs)
    debris = np.zeros_like(raw_scan)
    for _ in range(6):
        r, c = np.random.randint(20, 230, 2)
        debris[r, c] = 1.0
    debris = gaussian(debris, sigma=2.5) * 180.0

    final_highres = raw_scan + tilt + highres_lines + debris + np.random.normal(0, 2.0, (canvas_size, canvas_size))

    # ==========================================
    # 4. GENERATE CONTINUOUS 3x3 SURVEY GRID
    # ==========================================
    tile_h, tile_w = 64, 128
    full_h, full_w = tile_h * 3, tile_w * 3

    Y_grid, X_grid = np.mgrid[0:full_h, 0:full_w]
    macro_surface = 120.0 * (X_grid / full_w) + 60.0 * (Y_grid / full_h)
    macro_surface += np.random.normal(0, 1.5, (full_h, full_w))
    macro_surface += np.repeat(np.random.normal(0, 4.0, (full_h, 1)), full_w, axis=1)

    cy, cx = full_h // 2, full_w // 2
    
    # Central marker for survey scan rotated by 195 and mirrored to match the high-res
    cross_mask = np.zeros((80, 80))
    cross_mask[10:70, 37:43] = 150.0  # Vertical arm
    cross_mask[37:43, 20:60] = 150.0  # Horizontal arm
    cross_mask = rotate(np.fliplr(cross_mask), 195, resize=False)
    
    macro_surface[cy-40:cy+40, cx-40:cx+40] += cross_mask
    macro_surface = gaussian(macro_surface, sigma=1.5) * 1.0

    for r in range(3):
        for c in range(3):
            tile = macro_surface[r * tile_h : (r + 1) * tile_h, c * tile_w : (c + 1) * tile_w]
            tile_path = os.path.join(output_dir, f"Task_2.0_{r}.{c}_scan1.npy")
            np.save(tile_path, tile)

    highres_path = os.path.join(output_dir, "Task_3.6_synthetic_highres.npy")
    np.save(highres_path, final_highres)
    print(f"✅ Synthetic data saved successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--digits", type=str, default="1337")
    parser.add_argument("--output", type=str, default="data/raw/synthetic")
    args = parser.parse_args()
    generate_synthetic_data(args.output, args.digits)