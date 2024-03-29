import tensorflow as tf
from keras import Sequential, Input, Model
from keras.src.applications import VGG16
from keras.src.layers import Flatten, Dense, Conv2D, MaxPool2D, Dropout, SimpleRNN, Reshape, LSTM
from tensorflow import keras
from tensorflow.keras import layers
from keras.applications.vgg16 import VGG16, preprocess_input
class Modelsc:
    def Lstm(self):
        input_shape = (224, 224, 3)
        model=Sequential()
        model.add(Reshape((input_shape[0]*input_shape[1],input_shape[2]),input_shape=input_shape))
        model.add(LSTM(16))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(4, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model