import tensorflow as tf
from tensorflow.keras import layers


class Modelsc:
    def sgdx(self):
        model = tf.keras.Sequential()
        model.add(layers.Flatten(input_shape=(224, 224, 3)))  # Flatten layer to convert input to 1D

        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.BatchNormalization())

        model.add(layers.Dense(32, activation='relu'))  # Additional dense layer for complexity
        model.add(layers.Dropout(0.5))

        model.add(
            layers.Dense(4, activation='softmax'))  # Output layer with softmax activation for multiclass classification

        model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

        return model
