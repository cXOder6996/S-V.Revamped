"""
Inference module for SkinVision.
Handles model loading and prediction generation.
"""

import os
import numpy as np
import tensorflow as tf

from config import (
    CLASS_KEYS,
    CLASS_INFO,
    NUM_CLASSES,
    DEFAULT_MODEL_PATH,
    IMG_SIZE
)


def load_model(model_path: str = None) -> tf.keras.Model:
    """
    Load the trained TensorFlow/Keras model.
    
    Args:
        model_path: Path to the .keras model file. If None, uses DEFAULT_MODEL_PATH.
        
    Returns:
        Loaded tf.keras.Model
        
    Raises:
        FileNotFoundError: If the model file does not exist.
    """
    if model_path is None:
        model_path = DEFAULT_MODEL_PATH
        
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found at {model_path}. "
            f"Please ensure you have trained the model or downloaded the pretrained weights."
        )
        
    # Load model
    model = tf.keras.models.load_model(model_path)
    return model


def predict(model: tf.keras.Model, preprocessed_image: np.ndarray) -> np.ndarray:
    """
    Generate class probabilities for a preprocessed image.
    
    Args:
        model: Loaded Keras model
        preprocessed_image: Image array of shape (1, IMG_SIZE, IMG_SIZE, 3) 
                            with raw [0, 255] float values.
                            
    Returns:
        1D array of shape (NUM_CLASSES,) containing prediction probabilities.
        
    Raises:
        ValueError: If the input shape is incorrect.
    """
    expected_shape = (1, IMG_SIZE[0], IMG_SIZE[1], 3)
    if preprocessed_image.shape != expected_shape:
        raise ValueError(
            f"Expected input shape {expected_shape}, but got {preprocessed_image.shape}."
        )
        
    # Model returns shape (1, NUM_CLASSES)
    probabilities = model.predict(preprocessed_image, verbose=0)
    
    return probabilities[0]


class PredictionItem(dict):
    """Dictionary that also supports tuple indexing [0] -> class_key, [1] -> probability."""
    def __init__(self, class_key: str, class_name: str, probability: float):
        super().__init__(class_key=class_key, class_name=class_name, probability=probability)
        self.class_key = class_key
        self.class_name = class_name
        self.probability = probability
        
    def __getitem__(self, key):
        if key == 0:
            return self.class_key
        elif key == 1:
            return self.probability
        return super().__getitem__(key)


def get_top_k(probabilities: np.ndarray, classes: list = None, k: int = 3) -> list:
    """
    Get the top k predictions sorted by probability descending.
    
    Args:
        probabilities: 1D array of prediction probabilities
        classes: Optional list of class names/keys
        k: Number of top predictions to return
        
    Returns:
        List of PredictionItem objects containing class_key, class_name, and probability.
    """
    if isinstance(classes, int) and k == 3:
        k = classes
        classes = None
        
    class_list = classes if classes is not None else CLASS_KEYS
    if len(probabilities) != len(class_list):
        raise ValueError(f"Expected probabilities array of length {len(class_list)}, got {len(probabilities)}")
        
    top_k_indices = np.argsort(probabilities)[-k:][::-1]
    
    results = []
    for idx in top_k_indices:
        class_key = class_list[idx]
        class_name = CLASS_INFO.get(class_key, class_key)
        prob = float(probabilities[idx])
        
        results.append(PredictionItem(
            class_key=class_key,
            class_name=class_name,
            probability=prob
        ))
        
    return results
