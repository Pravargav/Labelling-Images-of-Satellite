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

train_directory = r'C:\Users\dell\PycharmProjects\pythonProject\Skill\genData\train'
valid_directory = r'C:\Users\dell\PycharmProjects\pythonProject\Skill\genData\valid'

cloud_directory = r'C:\Users\dell\PycharmProjects\pythonProject\Skill\data\cloudy'
cloud_train_directory = r'C:\Users\dell\PycharmProjects\pythonProject\Skill\genData\train\cloudy'
cloud_valid_directory = r'C:\Users\dell\PycharmProjects\pythonProject\Skill\genData\valid\cloudy'

water_directory = r'C:\Users\dell\PycharmProjects\pythonProject\Skill\data\water'
water_train_directory = r'C:\Users\dell\PycharmProjects\pythonProject\Skill\genData\train\water'
water_valid_directory = r'C:\Users\dell\PycharmProjects\pythonProject\Skill\genData\valid\water'

green_directory = r'C:\Users\dell\PycharmProjects\pythonProject\Skill\data\green_area'
green_train_directory = r'C:\Users\dell\PycharmProjects\pythonProject\Skill\genData\train\green'
green_valid_directory = r'C:\Users\dell\PycharmProjects\pythonProject\Skill\genData\valid\green'

desert_directory = r'C:\Users\dell\PycharmProjects\pythonProject\Skill\data\desert'
desert_train_directory = r'C:\Users\dell\PycharmProjects\pythonProject\Skill\genData\train\desert'
desert_valid_directory = r'C:\Users\dell\PycharmProjects\pythonProject\Skill\genData\valid\desert'

cloud_image_files = [f for f in os.listdir(cloud_directory) if
                     f.lower().endswith(('.jpg', '.jpeg'))]
# Specify a new name format
# You can customize the format according to your needs
# In this example, we're using a base name and appending an incremental number
new_name_base = 'cloudy'
start_index = 1

# Loop through the image files and rename them
for old_name in cloud_image_files:
    # Determine the file extension
    file_ext = os.path.splitext(old_name)[-1]

    # Create the new file name with the desired format
    new_name = f'{new_name_base}_{start_index:04d}{file_ext}'

    # Construct the old and new file paths
    old_path = os.path.join(cloud_directory, old_name)
    new_path = os.path.join(cloud_directory, new_name)

    # Rename the file
    os.rename(old_path, new_path)

    # Increment the start index
    start_index += 1

water_image_files = [f for f in os.listdir(water_directory) if
                     f.lower().endswith(('.jpg', '.jpeg'))]
# Specify a new name format
# You can customize the format according to your needs
# In this example, we're using a base name and appending an incremental number
new_name_base = 'water'
start_index = 1

# Loop through the image files and rename them
for old_name in water_image_files:
    # Determine the file extension
    file_ext = os.path.splitext(old_name)[-1]

    # Create the new file name with the desired format
    new_name = f'{new_name_base}_{start_index:04d}{file_ext}'

    # Construct the old and new file paths
    old_path = os.path.join(water_directory, old_name)
    new_path = os.path.join(water_directory, new_name)

    # Rename the file
    os.rename(old_path, new_path)

    # Increment the start index
    start_index += 1

desert_image_files = [f for f in os.listdir(desert_directory) if
                      f.lower().endswith(('.jpg', '.jpeg'))]
# Specify a new name format
# You can customize the format according to your needs
# In this example, we're using a base name and appending an incremental number
new_name_base = 'desert'
start_index = 1

# Loop through the image files and rename them
for old_name in desert_image_files:
    # Determine the file extension
    file_ext = os.path.splitext(old_name)[-1]

    # Create the new file name with the desired format
    new_name = f'{new_name_base}_{start_index:04d}{file_ext}'

    # Construct the old and new file paths
    old_path = os.path.join(desert_directory, old_name)
    new_path = os.path.join(desert_directory, new_name)

    # Rename the file
    os.rename(old_path, new_path)

    # Increment the start index
    start_index += 1

green_image_files = [f for f in os.listdir(green_directory) if
                     f.lower().endswith(('.jpg', '.jpeg'))]
# Specify a new name format
# You can customize the format according to your needs
# In this example, we're using a base name and appending an incremental number
new_name_base = 'green_area'
start_index = 1

# Loop through the image files and rename them
for old_name in green_image_files:
    # Determine the file extension
    file_ext = os.path.splitext(old_name)[-1]

    # Create the new file name with the desired format
    new_name = f'{new_name_base}_{start_index:04d}{file_ext}'

    # Construct the old and new file paths
    old_path = os.path.join(green_directory, old_name)
    new_path = os.path.join(green_directory, new_name)

    # Rename the file
    os.rename(old_path, new_path)

    # Increment the start index
    start_index += 1


os.makedirs(train_directory, exist_ok=True)
os.makedirs(valid_directory, exist_ok=True)

cloud_train_files, cloud_valid_files = train_test_split(cloud_image_files, test_size=0.2, random_state=42)
for file in cloud_train_files:
    source_file_path = os.path.join(cloud_directory, file)
    destination_file_path = os.path.join(cloud_train_directory, file)
    shutil.copy(source_file_path, destination_file_path)

for file in cloud_valid_files:
    source_file_path = os.path.join(cloud_directory, file)
    destination_file_path = os.path.join(cloud_valid_directory, file)
    shutil.copy(source_file_path, destination_file_path)

water_train_files, water_valid_files = train_test_split(water_image_files, test_size=0.2, random_state=42)
for file in water_train_files:
    source_file_path = os.path.join(water_directory, file)
    destination_file_path = os.path.join(water_train_directory, file)
    shutil.copy(source_file_path, destination_file_path)

for file in water_valid_files:
    source_file_path = os.path.join(water_directory, file)
    destination_file_path = os.path.join(water_valid_directory, file)
    shutil.copy(source_file_path, destination_file_path)

desert_train_files, desert_valid_files = train_test_split(desert_image_files, test_size=0.2, random_state=42)
for file in desert_train_files:
    source_file_path = os.path.join(desert_directory, file)
    destination_file_path = os.path.join(desert_train_directory, file)
    shutil.copy(source_file_path, destination_file_path)

for file in desert_valid_files:
    source_file_path = os.path.join(desert_directory, file)
    destination_file_path = os.path.join(desert_valid_directory, file)
    shutil.copy(source_file_path, destination_file_path)

green_train_files, green_valid_files = train_test_split(green_image_files, test_size=0.2, random_state=42)
for file in green_train_files:
    source_file_path = os.path.join(green_directory, file)
    destination_file_path = os.path.join(green_train_directory, file)
    shutil.copy(source_file_path, destination_file_path)

for file in green_valid_files:
    source_file_path = os.path.join(green_directory, file)
    destination_file_path = os.path.join(green_valid_directory, file)
    shutil.copy(source_file_path, destination_file_path)

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
    fill_mode='nearest'

)

train_generator = train_imagenerator.flow_from_directory(
    train_directory,
    target_size=(224,224),
    batch_size=20,
    class_mode='categorical',
#     classes=class_labels,
    shuffle=True
)

val_imagenerator = ImageDataGenerator(rescale=1.0/255)

validation_generator = val_imagenerator.flow_from_directory(
    valid_directory,
    target_size=(224,224),
    batch_size=20,
    class_mode='categorical',
    shuffle=True
)

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
batch_size = 28
image_size = (224, 224)

# Create ImageDataGenerator
train_imagenerator = ImageDataGenerator(
    rescale=1.0/255,   # Normalize pixel values to [0, 1]
    # You can add other data augmentation options here
)

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

model = Sequential()
pretrained_base_model = keras.applications.ResNet50(
    weights='imagenet',
    include_top=False, # Drop imagenet classifier on the top
    input_shape=(224,224,3),
    pooling = 'avg',
    classes=100

)

for layer in pretrained_base_model.layers:
  layer.trainable = False

model.add(pretrained_base_model)
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5)) # to prevent overfitting
model.add(Dense(100, activation='softmax'))

from tensorflow.keras.optimizers import Adam
model.compile(optimizer=Adam(),
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])

pretrained_base_model.summary()

model.summary()

# Define the input shape
inputs = pretrained_base_model.input

# stack the inputs to pretrained model and set training to false
x = pretrained_base_model.output


# Add drop out layer

x = tf.keras.layers.Dropout(0.4)(x)

# Last output dense layer with 1 unit and sigmoid
output = tf.keras.layers.Dense(100, activation='softmax')(x)

# Build a model
resnet=Model(inputs=inputs, outputs=output)

pretrained_base_model.trainable = True
print("Number of layers in the base Mnet model: ", len(pretrained_base_model.layers))
for layer in pretrained_base_model.layers:
  layer.trainable = True
for layer in pretrained_base_model.layers[:100]:
    layer.trainable = False

resnet.compile(
    optimizer='rmsprop',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']

)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=2,
    restore_best_weights=True
)

# Train the model with mini-batch gradient descent, calculate loss, and monitor metrics
resnet = resnet.fit(
    train_data,                   # Training data
    epochs=5,                # Number of training epochs
    batch_size=batch_size,        # Batch size
    validation_data=val_data,     # Validation data
    callbacks=[early_stopping],   # Early stopping callback
)

plt.figure(figsize=(10,3))
plt.plot(resnet.history['accuracy'])
plt.plot(resnet.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()