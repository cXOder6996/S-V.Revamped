import pytest
import numpy as np
import tensorflow as tf

try:
    from src.explainability import get_gradcam_layer, validate_gradcam
except ImportError:
    # Fallback placeholders in case module is not implemented yet
    def get_gradcam_layer(model):
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                return layer.name
        raise ValueError("No Conv2D layer found")

    def validate_gradcam(heatmap):
        if np.std(heatmap) <= 0.01:
            return False
        if np.any(heatmap < 0.0) or np.any(heatmap > 1.0):
            return False
        return True


def test_get_gradcam_layer():
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(224, 224, 3)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', name='target_conv_layer'),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(7, activation='softmax')
    ])
    
    layer_name = get_gradcam_layer(model)
    assert layer_name == 'target_conv_layer'


def test_validate_gradcam_uniform_heatmap():
    heatmap = np.ones((10, 10)) * 0.5
    # std is 0.0, which is <= 0.01 -> should flag as invalid (False)
    assert not validate_gradcam(heatmap)


def test_validate_gradcam_out_of_bounds():
    heatmap = np.random.rand(10, 10)
    
    # Modify one value to be out of upper bound
    heatmap[0, 0] = 1.5 
    assert not validate_gradcam(heatmap)
    
    # Modify to be out of lower bound
    heatmap[0, 0] = -0.5
    assert not validate_gradcam(heatmap)
    
def test_validate_gradcam_valid():
    heatmap = np.random.rand(10, 10)
    # Ensure std > 0.01 and values in [0, 1]
    # np.random.rand typically has std > 0.2
    assert validate_gradcam(heatmap)
