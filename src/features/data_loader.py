#%%
import os 
import keras
import numpy as np
# %%
def load_and_normalize_dataset():
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    x_train=x_train/255.0
    x_test=x_test/255.0
    return  (x_train, y_train), (x_test, y_test) 
# %%
def preprocess_incoming_image(image_array):
    scaled_image= image_array/255.0
    return np.expand_dims(image_array,axis=3)