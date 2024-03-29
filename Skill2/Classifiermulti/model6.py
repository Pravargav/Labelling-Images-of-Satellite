import tensorflow as tf
from keras import Sequential, Input, Model
from keras.src.applications import VGG16
from keras.src.layers import Flatten, Dense, Conv2D, MaxPool2D, Dropout, SimpleRNN, Reshape
from tensorflow import keras
from tensorflow.keras import layers
from keras.applications.vgg16 import VGG16, preprocess_input
class Modelsc:
    def rnn_model(self):
        input_shape=(224,224,3)
        model = Sequential()
        model.add(Reshape((input_shape[0] * input_shape[1], input_shape[2]), input_shape=input_shape))
        # Add SimpleRNN layer
        model.add(SimpleRNN(128))

        # Add fully connected layers
        model.add(Dense(4096, activation='relu'))
        model.add(Dense(1072, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(4, activation='softmax'))
        model.compile(optimizer='adam',
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])
        return model