
import model2 as mc


from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Directories
train_directory = r'C:\Users\dell\PycharmProjex\dlSkill\Skill\genData\train'
valid_directory = r'C:\Users\dell\PycharmProjex\dlSkill\Skill\genData\valid'

# Image data preprocessing
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
)

validation_datagen = ImageDataGenerator(rescale=1.0 / 255)

batch_size = 32

train_generator = train_datagen.flow_from_directory(
    train_directory,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)

validation_generator = validation_datagen.flow_from_directory(
    valid_directory,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)

# Sample CNN model
mc2=mc.Modelsc()
model2=mc2.sgdx()

# Display model summary
model2.summary()

# Random Mini-Batch Gradient Descent with evaluation
# Random Mini-Batch Gradient Descent with evaluation
num_epochs = 12
num_steps_train = len(train_generator)
num_steps_valid = len(validation_generator)

train_accs = []
valid_accs = []

import numpy as np

for epoch in range(num_epochs):
    train_generator.reset()
    validation_generator.reset()

    train_preds = []

    # Shuffle indices for training data
    Shuffle_indices = np.random.permutation(len(train_generator))

    # Training step
    for step in range(num_steps_train):
        # Get batch data with shuffled indices
        X_batch, y_batch = train_generator[Shuffle_indices[step]]

        train_loss, train_acc = model2.train_on_batch(X_batch, y_batch)

    # Validation step
    valid_acc = model2.evaluate_generator(validation_generator)[1]  # Get validation accuracy
    valid_accs.append(valid_acc)

    train_accs.append(train_acc)

    print(f"Epoch {epoch + 1}/{num_epochs} - Train Accuracy: {train_acc}, Validation Accuracy: {valid_acc}")


