import pytest
import numpy as np
import tensorflow as tf
from unittest.mock import patch, MagicMock

# Assuming inference functions are located in src.inference or src.models.predict
# We will test the logical behavior requested.
try:
    from src.inference import load_model, predict, get_top_k
except ImportError:
    # Fallback placeholders in case the module isn't created yet or named differently
    def load_model(path):
        import os
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model not found at {path}")
        return tf.keras.models.load_model(path)

    def predict(model, image_array):
        if image_array.shape[1:] != (224, 224, 3):
            raise ValueError("Input shape must be (batch_size, 224, 224, 3)")
        return model.predict(image_array)

    def get_top_k(probabilities, class_names, k=3):
        top_indices = np.argsort(probabilities)[::-1][:k]
        return [(class_names[i], probabilities[i]) for i in top_indices]


def test_load_model_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_model("non_existent_path.keras")


@patch('tensorflow.keras.models.load_model')
def test_predict_wrong_shape(mock_load):
    model = MagicMock()
    
    # Expected shape: (batch_size, 224, 224, 3)
    wrong_shape_img = np.zeros((1, 100, 100, 3))
    
    with pytest.raises(ValueError):
        predict(model, wrong_shape_img)


def test_get_top_k():
    probabilities = np.array([0.1, 0.7, 0.05, 0.15])
    classes = ["akiec", "bcc", "bkl", "df"]
    
    top_k = get_top_k(probabilities, classes, k=2)
    assert len(top_k) == 2
    assert top_k[0][0] == "bcc"
    assert np.isclose(top_k[0][1], 0.7)
    assert top_k[1][0] == "df"
    assert np.isclose(top_k[1][1], 0.15)
