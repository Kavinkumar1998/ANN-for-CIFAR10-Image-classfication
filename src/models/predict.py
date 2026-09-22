from __future__ import annotations

import os

import numpy as np
from keras.models import load_model

from src.features.data_loader import preprocess_incoming_image
from src.models import config


class Predictor:
    def __init__(self, model_path: str | None = None):
        model_path = model_path or config.MODEL_PATH
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found at {model_path}. Train the model first."
            )

        self.model = load_model(model_path)

    def predict(self, image_array):
        image = preprocess_incoming_image(image_array)

        if image.shape[1:] != config.INPUT_SHAPE:
            raise ValueError(
                f"Input image must have shape {config.INPUT_SHAPE}, but got {image.shape[1:]}"
            )

        prediction = self.model.predict(image, verbose=0)
        class_index = int(np.argmax(prediction[0]))
        return config.LABELS[class_index]


if __name__ == "__main__":
    predictor = Predictor()
    print("Predictor loaded successfully.")
