"""
Explainability module for the SkinVision project.
Provides Grad-CAM implementation for visualizing model predictions.
"""

import logging
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.cm as cm

try:
    import cv2
except ImportError:
    cv2 = None

logger = logging.getLogger(__name__)

def _is_4d_layer(layer) -> bool:
    shape = None
    if hasattr(layer, 'output_shape') and layer.output_shape is not None:
        shape = layer.output_shape[0] if isinstance(layer.output_shape, list) else layer.output_shape
    elif hasattr(layer, 'output') and hasattr(layer.output, 'shape') and layer.output.shape is not None:
        shape = layer.output.shape
    return shape is not None and len(shape) == 4


def _is_conv_layer(layer) -> bool:
    type_name = type(layer).__name__.lower()
    return 'conv' in type_name or 'conv' in layer.name.lower()


def get_gradcam_layer(model: tf.keras.Model) -> str:
    """
    Automatically find the last convolutional layer in the model.
    Walks model.layers backwards, finding the last layer whose output has 4 dimensions.
    If the model has a nested base model (Sequential wrapping EfficientNetB0), walks into the inner model.
    
    Args:
        model: The trained TensorFlow Keras model.
        
    Returns:
        str: The name of the last convolutional layer.
        
    Raises:
        ValueError: If no convolutional layer is found in the model.
    """
    # First pass: look specifically for convolutional layers with 4D output
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.Model):
            for inner_layer in reversed(layer.layers):
                if _is_conv_layer(inner_layer) and _is_4d_layer(inner_layer):
                    logger.info(f"Selected Grad-CAM layer: {inner_layer.name} from nested model {layer.name}")
                    return inner_layer.name
        
        if _is_conv_layer(layer) and _is_4d_layer(layer):
            logger.info(f"Selected Grad-CAM layer: {layer.name}")
            return layer.name

    # Second pass fallback: any layer with 4D output
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.Model):
            for inner_layer in reversed(layer.layers):
                if _is_4d_layer(inner_layer):
                    logger.info(f"Selected Grad-CAM layer: {inner_layer.name} from nested model {layer.name}")
                    return inner_layer.name
        
        if _is_4d_layer(layer):
            logger.info(f"Selected Grad-CAM layer: {layer.name}")
            return layer.name
                
    raise ValueError("No convolutional layer found in the model.")


def generate_gradcam(model: tf.keras.Model, image: np.ndarray, class_index: int = None, layer_name: str = None) -> np.ndarray:
    """
    Generates a Grad-CAM heatmap for a given image and model.
    
    Args:
        model: The trained TensorFlow Keras model.
        image: The input image array. Should match the training preprocessing (e.g., raw [0, 255] float input).
        class_index: The class index to generate the heatmap for. If None, uses the model's top prediction.
        layer_name: The name of the convolutional layer to use. If None, automatically finds it.
        
    Returns:
        np.ndarray: The generated Grad-CAM heatmap as a float32 array in [0, 1], 
                    resized to the input image spatial dimensions.
    """
    if layer_name is None:
        layer_name = get_gradcam_layer(model)
        
    # Find the target layer and whether it is inside a nested model
    target_layer = None
    base_model = None
    
    for layer in model.layers:
        if layer.name == layer_name:
            target_layer = layer
            break
        if isinstance(layer, tf.keras.Model):
            try:
                target_layer = layer.get_layer(layer_name)
                base_model = layer
                break
            except ValueError:
                pass
                
    if target_layer is None:
        raise ValueError(f"Layer {layer_name} not found in the model.")
        
    # Prepare the input. Ensure it has a batch dimension.
    if len(image.shape) == 3:
        img_array = np.expand_dims(image, axis=0)
    else:
        img_array = image
        
    with tf.GradientTape() as tape:
        if base_model is not None:
            # The target layer is inside a nested base model
            inner_grad_model = tf.keras.Model(
                inputs=base_model.inputs[0] if len(base_model.inputs) == 1 else base_model.inputs,
                outputs=[target_layer.output, base_model.output]
            )
            conv_outputs, inner_preds = inner_grad_model(img_array)
            tape.watch(conv_outputs)
            
            # Apply remaining layers to get final predictions
            x = inner_preds
            started = False
            for layer in model.layers:
                if started:
                    x = layer(x)
                if layer is base_model:
                    started = True
            preds = x
        else:
            # Target layer is directly in the top-level model
            grad_model = tf.keras.Model(
                inputs=model.inputs[0] if len(model.inputs) == 1 else model.inputs,
                outputs=[target_layer.output, model.output]
            )
            conv_outputs, preds = grad_model(img_array)
            tape.watch(conv_outputs)
                
        if class_index is None:
            target_class = tf.argmax(preds[0])
        else:
            target_class = class_index
            
        class_channel = preds[:, target_class]
    
    # Compute gradients of the predicted class score with respect to the conv layer's output
    grads = tape.gradient(class_channel, conv_outputs)
    
    # Global average pooling of the gradients to get feature weights
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Multiply each channel in the feature map array by its weight
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    # Apply ReLU (discard negative values)
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.math.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val
        
    heatmap = heatmap.numpy()
    
    # Resize the heatmap to match the spatial dimensions of the input image
    spatial_dims = (img_array.shape[2], img_array.shape[1])  # (W, H)
    if cv2 is not None:
        heatmap = cv2.resize(heatmap, spatial_dims)
    else:
        pil_hm = Image.fromarray(heatmap).resize(spatial_dims, Image.Resampling.BILINEAR)
        heatmap = np.array(pil_hm)
    
    return heatmap.astype(np.float32)


class ValidationResult(tuple):
    """Tuple subclass whose boolean truth value matches its first element (is_valid)."""
    def __bool__(self):
        return bool(self[0]) if len(self) > 0 else False


def validate_gradcam(heatmap: np.ndarray, image: np.ndarray = None) -> tuple[bool, list[str]]:
    """
    Validates a generated Grad-CAM heatmap to ensure it contains meaningful information.
    
    Args:
        heatmap: The generated heatmap array.
        image: Optional original image array.
        
    Returns:
        tuple[bool, list[str]]: A tuple containing a boolean indicating if the heatmap is valid, 
                                and a list of warning messages if any.
    """
    warnings = []
    
    # Check heatmap values are in [0, 1]
    if np.min(heatmap) < -1e-5 or np.max(heatmap) > 1.0 + 1e-5:
        warnings.append(f"Heatmap values are out of expected [0, 1] range: min={np.min(heatmap):.4f}, max={np.max(heatmap):.4f}.")
        
    # Check if heatmap is not uniform
    if np.std(heatmap) <= 0.01:
        warnings.append("Heatmap is nearly uniform (std <= 0.01), indicating the model may not be focusing on any specific features.")
        
    if image is not None:
        # Handle batch dimension in image if present
        if len(image.shape) == 4:
            h, w = image.shape[1:3]
        else:
            h, w = image.shape[:2]
            
        # Check heatmap shape matches image spatial dims
        if heatmap.shape != (h, w):
            warnings.append(f"Heatmap shape {heatmap.shape} does not match image spatial dimensions {(h, w)}.")
            
        # Check heatmap has some activation in the center region (center 50% of image)
        center_h_start, center_h_end = int(h * 0.25), int(h * 0.75)
        center_w_start, center_w_end = int(w * 0.25), int(w * 0.75)
        center_region = heatmap[center_h_start:center_h_end, center_w_start:center_w_end]
        
        if np.max(center_region) < 0.2:
            warnings.append("Low activation in the center region of the image. The model might not be focusing on the primary lesion.")
        
    is_valid = len(warnings) == 0
    return ValidationResult((is_valid, warnings))


def overlay_gradcam(original_image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """
    Overlays a Grad-CAM heatmap on the original image.
    
    Args:
        original_image: The original image array (H, W, 3).
        heatmap: The generated heatmap array (H, W) with values in [0, 1].
        alpha: The blending weight for the heatmap overlay.
        
    Returns:
        np.ndarray: The blended image as a uint8 RGB array.
    """
    # Remove batch dim if present
    if len(original_image.shape) == 4:
        original_image = original_image[0]
        
    # Convert original image to uint8
    if original_image.dtype != np.uint8:
        if np.max(original_image) <= 1.0:
            img_uint8 = np.clip(original_image * 255, 0, 255).astype(np.uint8)
        else:
            img_uint8 = np.clip(original_image, 0, 255).astype(np.uint8)
    else:
        img_uint8 = original_image.copy()

    if cv2 is not None:
        # Convert heatmap to uint8
        heatmap_uint8 = np.uint8(255 * heatmap)
        # Apply colormap (generates BGR)
        colormap_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        # Convert BGR to RGB
        colormap_rgb = cv2.cvtColor(colormap_bgr, cv2.COLORMAP_BGR2RGB)
        # Blend the images
        overlayed = cv2.addWeighted(img_uint8, 1 - alpha, colormap_rgb, alpha, 0)
    else:
        # Pure matplotlib / numpy colormap and blending fallback
        colormap_rgb = (cm.jet(np.clip(heatmap, 0.0, 1.0))[:, :, :3] * 255).astype(np.uint8)
        overlayed = np.clip(
            img_uint8.astype(np.float32) * (1.0 - alpha) + colormap_rgb.astype(np.float32) * alpha,
            0,
            255
        ).astype(np.uint8)
        
    return overlayed
