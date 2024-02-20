import numpy as np
import pandas as pd
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, GlobalAveragePooling2D, Dropout, Flatten, BatchNormalization
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image
import shutil
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
import model as mC

train_directory = r'C:\Users\dell\PycharmProjects\dlSkill\Skill\genData2\train'
valid_directory = r'C:\Users\dell\PycharmProjects\dlSkill\Skill\genData2\valid'

cloud_directory = r'C:\Users\dell\PycharmProjects\dlSkill\Skill\data\cloudy'

cloud_train_directory = r'C:\Users\dell\PycharmProjects\dlSkill\Skill\genData2\train\cloudy'
cloud_valid_directory = r'C:\Users\dell\PycharmProjects\dlSkill\Skill\genData2\valid\cloudy'

water_directory = r'C:\Users\dell\PycharmProjects\dlSkill\Skill\data\water'
green_directory = r'C:\Users\dell\PycharmProjects\dlSkill\Skill\data\green_area'
desert_directory = r'C:\Users\dell\PycharmProjects\dlSkill\Skill\data\desert'

non_cloud_train_directory = r'C:\Users\dell\PycharmProjects\dlSkill\Skill\genData2\train\noncloudy'
non_cloud_valid_directory = r'C:\Users\dell\PycharmProjects\dlSkill\Skill\genData2\valid\noncloudy'


cloud_image_files = [f for f in os.listdir(cloud_directory) if
                     f.lower().endswith(('.jpg', '.jpeg'))]
# Specify a new name format
# You can customize the format according to your needs


water_image_files = [f for f in os.listdir(water_directory) if
                     f.lower().endswith(('.jpg', '.jpeg'))]
# Specify a new name format
# You can customize the format according to your needs


desert_image_files = [f for f in os.listdir(desert_directory) if
                      f.lower().endswith(('.jpg', '.jpeg'))]
# Specify a new name format
# You can customize the format according to your needs
# In this example, we're using a base name and appending an incremental number


green_image_files = [f for f in os.listdir(green_directory) if
                     f.lower().endswith(('.jpg', '.jpeg'))]





os.makedirs(train_directory, exist_ok=True)
os.makedirs(valid_directory, exist_ok=True)




# Creating training image data generator
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_imagenerator = ImageDataGenerator(

    rescale=1.0 / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='wrap'

)

train_generator = train_imagenerator.flow_from_directory(
    train_directory,
    target_size=(224,224),
    batch_size=20,
    class_mode='binary',
    shuffle=True
)

val_imagenerator = ImageDataGenerator(rescale=1.0/255)

validation_generator = val_imagenerator.flow_from_directory(
    valid_directory,
    target_size=(224,224),
    batch_size=20,
    class_mode='binary',
    shuffle=True
)

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
batch_size = 30
image_size = (224, 224)

# Create ImageDataGenerator

# Create a tf.data.Dataset using image_dataset_from_directory
train_data = tf.keras.preprocessing.image_dataset_from_directory(
    train_directory,
    seed=45,
    shuffle=True,
    image_size=image_size,
    batch_size=batch_size
)

val_data = tf.keras.preprocessing.image_dataset_from_directory(
    valid_directory,
    seed=45,
    shuffle=False,
    image_size=image_size,
    batch_size=batch_size
)

class_names = train_data.class_names
print(class_names)

class_names = train_data.class_names

plt.figure(figsize=(12, 8))
for images, labels in train_data.take(1):
    for i in range(4):
        ax = plt.subplot(2, 2, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")

augmented_images, labels = train_generator.next()

plt.figure(figsize=(12, 8))
for i in range(min(6, augmented_images.shape[0])):
    ax = plt.subplot(2, 3, i + 1)
    plt.imshow(augmented_images[i])
    plt.title(int(np.argmax(labels[i])))  # Convert one-hot encoded label to integer category
    plt.axis("off")

class_names = train_data.class_names
print(class_names)

clsmC=mC.Modelsc()

model2=clsmC.adam()

h2 = model2.fit(
    train_data,                   # Training data
    epochs=12,                 # Number of training epochs
    batch_size=batch_size,        # Batch size
    validation_data=val_data,   # Early stopping callback
)

model3=clsmC.sgd()

h3 = model3.fit(
    train_data,                   # Training data
    epochs=4,                 # Number of training epochs
    batch_size=batch_size,        # Batch size
    validation_data=val_data,   # Early stopping callback
)

model4=clsmC.rmsprop()

h4 = model4.fit(
    train_data,                   # Training data
    epochs=4,                 # Number of training epochs
    batch_size=batch_size,        # Batch size
    validation_data=val_data,   # Early stopping callback
)

plt.figure(figsize=(10,3))
plt.plot(h2.history['accuracy'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()