import os

#Data configuration
IMG_ROWS = 32
IMG_COLS=32
CHANNELS=3
INPUT_SHAPE=(IMG_ROWS,IMG_COLS,CHANNELS)
NUM_CLASSES=10

#Hyperparameters
Batch_size=64
Epochs=10
Learning_rate=0.001
Optimizers="adam"
loss="sparse_categorical_crossentropy"
metrics=["categorical_accuracy"]

#CIFAR 10 Class Map
Labels={
    0:"airplane",1:"automobile",2:"bird",3:"cat",4:"deer",5:"dog",
    6:"frog",7:"horse",8:"ship",9:"truck"
}

#paths
Base_dir=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
Model_dir=os.path.join(Base_dir, "models")
Model_Path=os.path.join(Model_dir,"cifar10_production_model.h5")