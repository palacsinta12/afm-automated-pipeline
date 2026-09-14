import numpy as np
from scipy.ndimage import center_of_mass, shift
from skimage.transform import resize
from skimage.filters import gaussian
import tensorflow as tf
from tensorflow import keras

def preprocess_digit_for_mnist(digit_img: np.ndarray, target_inner_size: int = 20, canvas_size: int = 28):
    """
    Preprocesses a raw AFM digit crop into an MNIST-compatible array.
    """
    # CRITICAL FIX: AFM numpy arrays have row 0 at the physical bottom. 
    # MNIST models are trained with row 0 at the top.
    # An upside-down 2 mathematically forms a 5! We must flip it vertically.
    digit_img = np.flipud(digit_img)
    
    h, w = digit_img.shape
    
    # 1. Resize while preserving aspect ratio
    scale = target_inner_size / max(h, w)
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    resized = resize(digit_img, (new_h, new_w), anti_aliasing=True, preserve_range=True)
    
    # 2. Place roughly in the center of the canvas
    canvas = np.zeros((canvas_size, canvas_size), dtype=float)
    start_r = (canvas_size - new_h) // 2
    start_c = (canvas_size - new_w) // 2
    canvas[start_r:start_r + new_h, start_c:start_c + new_w] = resized
    
    # 3. Shift by Center of Mass (Fixes MNIST alignment expectations)
    cy, cx = center_of_mass(canvas)
    if not np.isnan(cy) and not np.isnan(cx):
        shift_y = (canvas_size / 2.0) - cy
        shift_x = (canvas_size / 2.0) - cx
        canvas = shift(canvas, shift=(shift_y, shift_x), order=1)
        
    # 4. Blur to mimic MNIST stroke gradients
    blurred = gaussian(canvas, sigma=0.6, preserve_range=True)
    
    # 5. Min-Max Normalize to [0, 1]
    img_min, img_max = blurred.min(), blurred.max()
    if img_max > img_min:
        normalized = (blurred - img_min) / (img_max - img_min)
    else:
        normalized = blurred
        
    return normalized

def load_and_predict(model_path: str, preprocessed_digits: list):
    """
    Loads the trained model and returns predictions.
    """
    model = keras.models.load_model(model_path)
    
    # Convert list of 2D arrays to (N, 28, 28, 1) batch tensor
    batch = np.array(preprocessed_digits, dtype=np.float32)
    batch = np.expand_dims(batch, axis=-1)
    
    # Predict
    # Predict (verbose=0 hides the progress bar)
    probabilities = model.predict(batch, verbose=0)
    labels = np.argmax(probabilities, axis=1)
    
    return labels, probabilities