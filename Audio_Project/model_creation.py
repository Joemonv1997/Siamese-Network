from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import *
print(tf.__version__)
print(keras.__version__)
from get_data import X_train,X_test,y_train,y_test
def build_base(input_shape):
    input=layers.Input(shape=input_shape)
    x=layers.Flatten()(input)
    x=layers.Dense(128,activation="relu")(x)
    x=layers.Dropout(0.1)(x)
    x=layers.Dense(128,activation="relu")(x)
    x=layers.Dropout(0.1)(x)
    x=layers.Dense(128,activation="relu")(x)
    return models.Model(input,x)


input_dim=X_train.shape[2::]
audio_a=layers.Input(shape=input_dim)
audio_b=layers.Input(shape=input_dim)
base_network=build_base(input_dim)
a=base_network(audio_a)
b=base_network(audio_b)


