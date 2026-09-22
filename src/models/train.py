# %%
import os
from keras.models import Sequential
from keras.layers import Flatten,Dense
import src.models.config as config
from src.features.data_loader import load_and_normalize_dataset
# %%
def build_cfr_ann():
    model = Sequential()
    model.add(Flatten(input_shape=config.INPUT_SHAPE))
    model.add(Dense(units=300,activation="relu"))
    model.add(Dense(units=150,activation="relu"))
    model.add(Dense(units=75,activation="relu"))
    model.add(Dense(units=36,activation="relu"))
    model.add(Dense(units=300,activation="softmax"))
    model.compile(optimizer=config.Optimizers,loss=config.loss,metrics=config.metrics)
    return model

def run_model_pipeline():
    x_train,y_train = load_and_normalize_dataset()[0]
    x_test,y_test = load_and_normalize_dataset()[1]
    print("compiling model")
    model=build_cfr_ann()
    print("fitting model")
    model.fit(x=x_train,y=y_train,batch_size=config.Batch_size,epochs=config.Epochs,validation_data=(x_test,y_test))
    model.summary()
    os.makedirs(os.path.dirname(config.Model_Path), exist_ok=True)
    print(f"Saving artifacts to: {config.Model_Path}")
    model.save(config.Model_Path)
    print("Pipeline sequence completed successfully.")

if __name__ == "__main__":
    run_model_pipeline()

# %%
