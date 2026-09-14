import numpy as np

def plane_correction(scan: np.ndarray, x: np.ndarray, y: np.ndarray, mask: np.ndarray = None):
    """
    Fits a plane to the unmasked regions of the scan and subtracts it.
    """
    if mask is None:
        mask = np.ones_like(scan, dtype=bool)
        
    X, Y = np.meshgrid(x, y)
    
    x_fit = X[mask]
    y_fit = Y[mask]
    z_fit = scan[mask]
    
    # Least squares plane fit: z = a*x + b*y + c
    A = np.column_stack((x_fit, y_fit, np.ones_like(x_fit)))
    coeffs, _, _, _ = np.linalg.lstsq(A, z_fit, rcond=None)
    
    fitted_plane = coeffs[0] * X + coeffs[1] * Y + coeffs[2]
    corrected_scan = scan - fitted_plane
    
    return corrected_scan, fitted_plane

def line_correction(scan: np.ndarray, xy: np.ndarray, p: int = 2, axis: int = 0, mask: np.ndarray = None):
    """
    Applies polynomial line-by-line correction to remove scanning stripe artifacts.
    """
    if mask is None:
        mask = np.ones_like(scan, dtype=bool)
        
    if axis == 1:
        scan = scan.T
        mask = mask.T
        
    corrected = np.zeros_like(scan)
    
    for i, (row, row_mask) in enumerate(zip(scan, mask)):
        if np.sum(row_mask) > p:
            # Normalize xy to [0, 1] for numerical stability
            xy_norm = (xy - xy.min()) / (xy.max() - xy.min() + 1e-12)
            poly_coeffs = np.polyfit(xy_norm[row_mask], row[row_mask], deg=p)
            fit_row = np.polyval(poly_coeffs, xy_norm)
            corrected[i] = row - fit_row
        else:
            corrected[i] = row
            
    if axis == 1:
        corrected = corrected.T
        
    return corrected