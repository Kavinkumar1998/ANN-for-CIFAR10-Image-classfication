from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "model"
MODEL_PATH = MODEL_DIR / "cifar10_production_model.h5"

# Data configuration
IMG_ROWS = 32
IMG_COLS = 32
CHANNELS = 3
INPUT_SHAPE = (IMG_ROWS, IMG_COLS, CHANNELS)
NUM_CLASSES = 10

# Hyperparameters
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.001
OPTIMIZER = "adam"
LOSS = "sparse_categorical_crossentropy"
METRICS = ["accuracy"]

# CIFAR-10 class map
LABELS = {
    0: "airplane",
    1: "automobile",
    2: "bird",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "frog",
    7: "horse",
    8: "ship",
    9: "truck",
}

# Backward-compatible aliases
Base_dir = str(PROJECT_ROOT)
Model_dir = str(MODEL_DIR)
Model_Path = str(MODEL_PATH)
Batch_size = BATCH_SIZE
Epochs = EPOCHS
Learning_rate = LEARNING_RATE
Optimizers = OPTIMIZER
loss = LOSS
metrics = METRICS
Labels = LABELS
