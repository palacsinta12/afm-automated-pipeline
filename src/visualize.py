import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os

def plot_scan_correction(raw_scan: np.ndarray, bg_mask: np.ndarray, corrected_scan: np.ndarray, save_dir: str):
    """Plots the raw scan, the detected background mask, and the flattened scan."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    im0 = axes[0].imshow(raw_scan, cmap="afmhot", origin="lower")
    axes[0].set_title("1. Raw High-Res Scan")
    plt.colorbar(im0, ax=axes[0], shrink=0.8)
    
    im1 = axes[1].imshow(bg_mask, cmap="gray", origin="lower")
    axes[1].set_title("2. Computed Background Mask")
    
    im2 = axes[2].imshow(corrected_scan, cmap="afmhot", origin="lower")
    axes[2].set_title("3. Flattened (Masked Correction)")
    plt.colorbar(im2, ax=axes[2], shrink=0.8)
    
    for ax in axes:
        ax.axis("off")
        
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "scan_correction.png"), dpi=150, bbox_inches="tight")
    plt.close()

def plot_extraction(aligned_scan: np.ndarray, aligned_mask: np.ndarray, save_dir: str):
    """Plots the geometrically aligned scan and its structural mask."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    im0 = axes[0].imshow(aligned_scan, cmap="afmhot", origin="lower")
    axes[0].set_title("Aligned Topography")
    
    im1 = axes[1].imshow(aligned_mask, cmap="gray", origin="lower")
    axes[1].set_title("Aligned Binary Mask")
    
    for ax in axes:
        ax.axis("off")
        
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "extraction.png"), dpi=150, bbox_inches="tight")
    plt.close()

def plot_cnn_results(preprocessed_digits: list, labels: list, probabilities: np.ndarray, save_dir: str):
    """Plots the preprocessed digits and the class probability heatmap."""
    n_digits = len(preprocessed_digits)

    fig, axes = plt.subplots(1, n_digits, figsize=(3 * n_digits, 3))
    if n_digits == 1:
        axes = [axes]

    for i in range(n_digits):
        axes[i].imshow(preprocessed_digits[i].squeeze(), cmap="gray", origin="upper")
        axes[i].set_title(f"Pred: {labels[i]}")
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "cnn_digits.png"), dpi=150, bbox_inches="tight")
    plt.close()

    pred_probs_safe = np.clip(probabilities, 1e-12, 1.0)
    fig, ax = plt.subplots(figsize=(10, 4))
    
    im = ax.imshow(pred_probs_safe, aspect="auto", norm=LogNorm(vmin=1e-12, vmax=1.0))
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(r"$P_{prediction}$")
    
    ax.set_xlabel("Digit Class")
    ax.set_ylabel("Crop Index")
    ax.set_title("CNN Class Probabilities (Logarithmic Scale)")
    
    ax.set_xticks(np.arange(10))
    ax.set_yticks(np.arange(n_digits))
    
    for i in range(n_digits):
        j = labels[i]
        rect_x = [j - 0.5, j + 0.5, j + 0.5, j - 0.5, j - 0.5]
        rect_y = [i - 0.5, i - 0.5, i + 0.5, i + 0.5, i - 0.5]
        ax.plot(rect_x, rect_y, color='red', linewidth=2)
        ax.text(j, i, str(j), ha="center", va="center", color="white", fontsize=12, fontweight="bold")
        
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "cnn_heatmap.png"), dpi=150, bbox_inches="tight")
    plt.close()

def plot_marker_detection(stitched_scan: np.ndarray, x0: float, y0: float, edge: np.ndarray, save_dir: str):
    """Plots the stitched survey scan with the detected centroid and the edge map."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].imshow(stitched_scan, cmap="afmhot", origin="lower")
    axes[0].plot(x0, y0, "bo", markersize=8, label="Detected Marker")
    axes[0].set_title(f"Stitched Survey Scan (Center: X={x0:.1f}, Y={y0:.1f})")
    axes[0].legend(loc="upper right")

    axes[1].imshow(edge, cmap="gray", origin="lower")
    axes[1].set_title("Sobel Gradient Edge Map")

    for ax in axes:
        ax.axis("off")
        
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "stitched_survey.png"), dpi=150, bbox_inches="tight")
    plt.close()