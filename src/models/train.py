from __future__ import annotations

import os

from keras.layers import Dense, Flatten, Input
from keras.models import Sequential

from src.features.data_loader import load_and_normalize_dataset
from src.models import config


def build_cfr_ann():
    model = Sequential(
        [
            Input(shape=config.INPUT_SHAPE),
            Flatten(),
            Dense(units=300, activation="relu"),
            Dense(units=150, activation="relu"),
            Dense(units=75, activation="relu"),
            Dense(units=36, activation="relu"),
            Dense(units=config.NUM_CLASSES, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=config.OPTIMIZER,
        loss=config.LOSS,
        metrics=config.METRICS,
    )
    return model


def run_model_pipeline():
    (x_train, y_train), (x_test, y_test) = load_and_normalize_dataset()

    print("Compiling model...")
    model = build_cfr_ann()

    print("Training model...")
    model.fit(
        x=x_train,
        y=y_train,
        batch_size=config.BATCH_SIZE,
        epochs=config.EPOCHS,
        validation_data=(x_test, y_test),
    )

    model.summary()

    os.makedirs(config.MODEL_DIR, exist_ok=True)
    print(f"Saving model to: {config.MODEL_PATH}")
    model.save(config.MODEL_PATH)
    print("Pipeline sequence completed successfully.")


if __name__ == "__main__":
    run_model_pipeline()

