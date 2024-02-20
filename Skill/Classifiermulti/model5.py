import tensorflow as tf
from keras import Sequential, Input, Model
from keras.src.applications import VGG16
from keras.src.layers import Flatten, Dense, Conv2D, MaxPool2D, Dropout
from tensorflow import keras
from tensorflow.keras import layers
from keras.applications.vgg16 import VGG16, preprocess_input
class Modelsc:

    def adam_with_regularization(self):
        model = tf.keras.Sequential()

        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.BatchNormalization())

        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.BatchNormalization())

        model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.BatchNormalization())

        model.add(layers.Flatten())

        model.add(layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(
            0.001)))  # Adding L2 regularization with 0.001 strength
        model.add(layers.Dropout(0.25))  # Adding dropout with 50% rate

        model.add(layers.Dense(4, activation='softmax'))  # Assuming 4 classes: cloudy, water, green, desert

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        return model





