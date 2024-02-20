import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
class Modelsc:
    def adam(self):
        model2 = tf.keras.Sequential([
            layers.Conv2D(64, (3, 3), padding="valid", input_shape=(224, 224, 3), activation='relu'),
            layers.MaxPooling2D(pool_size=(2, 2)),

            layers.Conv2D(16, (3, 3), padding="valid", activation='relu'),
            layers.MaxPooling2D(pool_size=(2, 2)),

            layers.Conv2D(32, (4, 4), padding="valid", activation='relu'),
            layers.MaxPooling2D(pool_size=(2, 2)),

            layers.Conv2D(128, (4, 4), padding="valid", activation='relu'),
            layers.MaxPooling2D(pool_size=(2, 2)),

            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.15),
            layers.Dense(2, activation='softmax')
        ])

        model2.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                      loss=tf.keras.losses.BinaryCrossentropy(),
                      metrics=[tf.keras.metrics.BinaryAccuracy()])
        return model2

    def sgd(self):
        model3 = tf.keras.Sequential([
            layers.Conv2D(64, (3, 3), padding="valid", input_shape=(224, 224, 3), activation='relu'),
            layers.MaxPooling2D(pool_size=(2, 2)),

            layers.Conv2D(16, (3, 3), padding="valid", activation='relu'),
            layers.MaxPooling2D(pool_size=(2, 2)),

            layers.Conv2D(32, (4, 4), padding="valid", activation='relu'),
            layers.MaxPooling2D(pool_size=(2, 2)),

            layers.Conv2D(128, (4, 4), padding="valid", activation='relu'),
            layers.MaxPooling2D(pool_size=(2, 2)),

            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.15),
            layers.Dense(1, activation='sigmoid')
        ])

        model3.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
                      loss=tf.keras.losses.BinaryCrossentropy(),
                      metrics=[tf.keras.metrics.BinaryAccuracy()])
        return model3

    def rmsprop(self):
        model4 = tf.keras.Sequential([
            layers.Conv2D(64, (3, 3), padding="valid", input_shape=(224, 224, 3), activation='relu'),
            layers.MaxPooling2D(pool_size=(2, 2)),

            layers.Conv2D(16, (3, 3), padding="valid", activation='relu'),
            layers.MaxPooling2D(pool_size=(2, 2)),

            layers.Conv2D(32, (4, 4), padding="valid", activation='relu'),
            layers.MaxPooling2D(pool_size=(2, 2)),

            layers.Conv2D(128, (4, 4), padding="valid", activation='relu'),
            layers.MaxPooling2D(pool_size=(2, 2)),

            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.15),
            layers.Dense(1, activation='sigmoid')
        ])

        model4.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.01),
                      loss=tf.keras.losses.BinaryCrossentropy(),
                      metrics=['accuracy'])
        return model4