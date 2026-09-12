import pytest
import numpy as np
from src.preprocessing import preprocess_for_inference, validate_image

def test_preprocess_for_inference_shape():
    dummy = np.zeros((300, 300, 3), dtype=np.uint8)
    output = preprocess_for_inference(dummy)
    assert output.shape == (1, 224, 224, 3)

def test_preprocess_for_inference_numerical_values():
    dummy = np.full((224, 224, 3), [100.0, 150.0, 200.0], dtype=np.float32)
    output = preprocess_for_inference(dummy)
    # Verifies NO normalization is applied (keeps raw [0, 255] float input for TF 2.20)
    assert np.allclose(output[0, 0, 0], [100.0, 150.0, 200.0])

def test_validate_image_formats():
    # grayscale (HxW)
    gray = np.zeros((100, 100), dtype=np.uint8)
    out = preprocess_for_inference(gray)
    assert out.shape[-1] == 3
    
    # grayscale (HxWx1)
    gray_1 = np.zeros((100, 100, 1), dtype=np.uint8)
    out_1 = preprocess_for_inference(gray_1)
    assert out_1.shape[-1] == 3
    
    # RGBA (HxWx4)
    rgba = np.zeros((100, 100, 4), dtype=np.uint8)
    out_rgba = preprocess_for_inference(rgba)
    assert out_rgba.shape[-1] == 3

def test_validate_image_quality():
    assert validate_image(None) is False
    
    small = np.zeros((32, 32, 3), dtype=np.uint8)
    assert validate_image(small) is False
    
    extreme = np.zeros((1000, 10, 3), dtype=np.uint8)
    assert validate_image(extreme) is False
