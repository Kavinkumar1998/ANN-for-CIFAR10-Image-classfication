from __future__ import annotations

import numpy as np
from keras import datasets


def load_and_normalize_dataset():
    """Load CIFAR-10 and normalize image pixels to [0, 1]."""
    (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0

    y_train = y_train.reshape(-1).astype(np.int32)
    y_test = y_test.reshape(-1).astype(np.int32)

    return (x_train, y_train), (x_test, y_test)


def preprocess_incoming_image(image_array):
    """Normalize and prepare a single RGB image for model input."""
    image_array = np.asarray(image_array, dtype=np.float32)
    if image_array.shape[-1] != 3:
        raise ValueError(f"Expected a 3-channel image, got shape {image_array.shape}.")

    image_array = image_array / 255.0
    return np.expand_dims(image_array, axis=0)
