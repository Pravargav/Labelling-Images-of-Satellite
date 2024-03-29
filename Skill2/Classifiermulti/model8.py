import tensorflow as tf
from keras import Sequential, Input, Model
from keras.src import losses
from keras.src.applications import VGG16
from keras.src.layers import Flatten, Dense, Conv2D, MaxPool2D, Dropout, SimpleRNN, Reshape, LSTM, MaxPooling2D, \
    UpSampling2D
from tensorflow import keras
from tensorflow.keras import layers
from keras.applications.vgg16 import VGG16, preprocess_input
class Modelsc:
    def Autoencoder(self):
        input = Input(shape=(224, 224, 3))
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(input)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        encoded = MaxPooling2D((2, 2), padding='same')(x)

        x = Conv2D(64, (3, 3), activation='relu', padding='same')(encoded)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        decoded = Conv2D(3, kernel_size=(3, 3), activation='sigmoid', padding='same')(x)

        autoencoder = Model(input, decoded)
        autoencoder.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

        return autoencoder


