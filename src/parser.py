import numpy as np

def read_scan_file(path: str, channel: int = 0):
    """
    Reads AFM data. Supports both proprietary .safm and synthetic .npy files.
    """
    if path.endswith(".safm"):
        import gwyfile
        gwy_container = gwyfile.load(path)
        meta = gwy_container[f"/{channel}/meta"]
        data = gwy_container[f"/{channel}/data"]
        
        z_data = np.array(data["data"])
        z_data = z_data.reshape(data["yres"], data["xres"])
        data["data"] = z_data
        
        return meta, data
        
    elif path.endswith(".npy"):
        # Handle synthetic data for the public GitHub repo
        z_data = np.load(path)
        
        # Mock metadata
        meta = {}
        data = {
            "data": z_data,
            "xres": z_data.shape[1],
            "yres": z_data.shape[0],
            "xreal": 1.0, 
            "yreal": 1.0
        }
        return meta, data
        
    else:
        raise ValueError("Unsupported file format. Use .safm or .npy")