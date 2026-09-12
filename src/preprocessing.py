"""
Image preprocessing module for the SkinVision project.
"""
try:
    import cv2
except ImportError:
    cv2 = None
import numpy as np
from PIL import Image

import sys
import os

# Import config — supports both package and standalone usage
try:
    from config import IMG_SIZE, MIN_IMAGE_SIZE, DOMAIN_NOTICE
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import IMG_SIZE, MIN_IMAGE_SIZE, DOMAIN_NOTICE

def preprocess_for_inference(image_array: np.ndarray) -> np.ndarray:
    """
    Preprocesses an image array for model inference.
    """
    # Convert to RGB if grayscale or RGBA
    if len(image_array.shape) == 2:
        if cv2 is not None:
            image = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
        else:
            image = np.stack([image_array] * 3, axis=-1)
    elif len(image_array.shape) == 3 and image_array.shape[2] == 1:
        image = np.repeat(image_array, 3, axis=-1)
    elif len(image_array.shape) == 3 and image_array.shape[2] == 4:
        if cv2 is not None:
            image = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)
        else:
            image = image_array[:, :, :3]
    else:
        image = image_array
        
    # Resize to IMG_SIZE if spatial dimensions differ
    if image.shape[:2] == (IMG_SIZE[0], IMG_SIZE[1]):
        image = image.astype(np.float32)
    else:
        if image.dtype != np.uint8:
            img_for_pil = np.clip(image, 0, 255).astype(np.uint8)
        else:
            img_for_pil = image
        pil_image = Image.fromarray(img_for_pil)
        pil_image = pil_image.resize(IMG_SIZE, Image.Resampling.LANCZOS)
        image = np.array(pil_image, dtype=np.float32)
    
    # TF 2.20 / Keras 3 EfficientNetB0 includes an internal Rescaling layer. 
    # The model expects raw [0, 255] float input. preprocess_input() is a no-op in this version.
    
    # Add batch dimension if needed
    if len(image.shape) == 3:
        return np.expand_dims(image, axis=0)
    return image


def validate_image(image_array: np.ndarray, return_message: bool = False):
    """
    Validates an image array before processing.
    
    Args:
        image_array: numpy array representing the image
        return_message: if True, returns tuple (bool, str). If False, returns bool.
    """
    if image_array is None:
        return (False, "Image is None.") if return_message else False
        
    if len(image_array.shape) not in [2, 3]:
        return (False, "Image must have 2 or 3 dimensions.") if return_message else False
        
    h, w = image_array.shape[:2]
    
    if h < MIN_IMAGE_SIZE[0] or w < MIN_IMAGE_SIZE[1]:
        return (False, f"Image size {w}x{h} is too small. Minimum size is {MIN_IMAGE_SIZE[0]}x{MIN_IMAGE_SIZE[1]}.") if return_message else False
        
    # Reasonable aspect ratio (neither dim > 10x the other)
    if h > 10 * w or w > 10 * h:
        return (False, f"Image aspect ratio is extreme ({w}x{h}).") if return_message else False
        
    return (True, "Valid image.") if return_message else True

def check_image_quality(image_array: np.ndarray) -> tuple[bool, list[str]]:
    """
    Checks image quality and returns warnings if poor quality.
    """
    passes_quality = True
    warnings = []
    
    if len(image_array.shape) == 2:
        img_gray = image_array
    elif len(image_array.shape) == 3 and image_array.shape[2] == 4:
        if cv2 is not None:
            img_gray = cv2.cvtColor(image_array, cv2.COLOR_RGBA2GRAY)
        else:
            img_gray = np.dot(image_array[:, :, :3].astype(np.float32), [0.2989, 0.5870, 0.1140])
    elif len(image_array.shape) == 3 and image_array.shape[2] == 3:
        if cv2 is not None:
            img_gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        else:
            img_gray = np.dot(image_array.astype(np.float32), [0.2989, 0.5870, 0.1140])
    else:
        img_gray = image_array
        
    # Check if mostly uniform (std <= 5 per channel)
    if np.std(image_array) <= 5:
        passes_quality = False
        warnings.append("Image is mostly uniform (lacks details).")
        
    # Check for reasonable brightness (mean between 10 and 245)
    mean_val = np.mean(image_array)
    if mean_val < 10 or mean_val > 245:
        passes_quality = False
        warnings.append(f"Image has extreme brightness levels (mean: {mean_val:.2f}).")
        
    # Check if not extremely blurry (Laplacian variance > 10)
    # Using gray scale image for Laplacian variance
    if cv2 is not None:
        laplacian_var = float(cv2.Laplacian(img_gray, cv2.CV_64F).var())
    else:
        try:
            from scipy.ndimage import laplace
            laplacian_var = float(laplace(img_gray).var())
        except Exception:
            # Approximate discrete Laplacian using numpy
            laplacian_var = float(np.var(img_gray[1:-1, 1:-1] * 4 - img_gray[:-2, 1:-1] - img_gray[2:, 1:-1] - img_gray[1:-1, :-2] - img_gray[1:-1, 2:]))
            
    if laplacian_var <= 10:
        passes_quality = False
        warnings.append(f"Image appears extremely blurry (Laplacian variance: {laplacian_var:.2f}).")
        
    if not passes_quality:
        warnings.append(f"Notice: {DOMAIN_NOTICE}")
        
    return passes_quality, warnings
