import tensorflow as tf
from keras import Sequential, Input, Model
from keras.src.applications import VGG16
from keras.src.layers import Flatten, Dense, Conv2D, MaxPool2D, Dropout
from tensorflow import keras
from tensorflow.keras import layers
from keras.applications.vgg16 import VGG16, preprocess_input
class Modelsc:
    def cnn_vgg(self):

        conv_base = VGG16(include_top=False,
                          weights='imagenet',
                          input_shape=(224, 224, 3))


        for layer in conv_base.layers:
            layer.trainable = False
        # Create a new 'top' of the model (i.e. fully-connected layers).
        # This is 'bootstrapping' a new top_model onto the pretrained layers.
        top_model = conv_base.output
        top_model = Flatten(name="flatten")(top_model)
        top_model = Dense(4096, activation='relu')(top_model)
        top_model = Dense(1072, activation='relu')(top_model)
        top_model = Dropout(0.2)(top_model)
        output_layer = Dense(4, activation='softmax')(top_model)

        # Group the convolutional base and new fully-connected layers into a Model object.
        model = Model(inputs=conv_base.input, outputs=output_layer)

        # Compiles the model for training.
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        return model



