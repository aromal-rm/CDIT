import os
import numpy as np
from keras.models import Sequential
from keras.layers import Cropping2D, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import shutil
import tensorflow as tf
from pathlib import Path

# === Step 1: Preprocessing ===
# Set paths for data directories
original_data_dir = "data"  # Replace with your dataset directory
base_dir = "data_preprocessed"  # New base directory for preprocessed data

# Create directories for training and validation datasets
if not os.path.exists(base_dir):
    os.makedirs(base_dir)
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "val")

for subdir in ["accepting", "not_accepting"]:
    os.makedirs(os.path.join(train_dir, subdir), exist_ok=True)
    os.makedirs(os.path.join(val_dir, subdir), exist_ok=True)

# Split the dataset into training and validation sets
for label in ["accepting", "not_accepting"]:
    src_dir = os.path.join(original_data_dir, label)
    all_files = os.listdir(src_dir)
    train_files, val_files = train_test_split(all_files, test_size=0.2, random_state=42)

    for file in train_files:
        shutil.copy(os.path.join(src_dir, file), os.path.join(train_dir, label, file))

    for file in val_files:
        shutil.copy(os.path.join(src_dir, file), os.path.join(val_dir, label, file))

print("Preprocessing complete and saved to 'data_preprocessed'.")

# === Step 2: Data Generators ===
# Training data generator
train_datagen = ImageDataGenerator(rescale=1.0 / 255)
train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode="binary",
    color_mode="grayscale",  # Convert to grayscale
)

# Validation data generator
val_datagen = ImageDataGenerator(rescale=1.0 / 255)
val_data = val_datagen.flow_from_directory(
    val_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode="binary",
    color_mode="grayscale",  # Convert to grayscale
)

# === Step 3: Model Definition ===
model = Sequential([
    Cropping2D(((20, 20), (0, 0)), input_shape=(128, 128, 1)),  # Grayscale input
    Conv2D(32, (3, 3), activation="relu"),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation="relu"),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(1, activation="sigmoid"),  # Binary classification
])

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

# === Step 4: Model Training ===
history = model.fit(
    train_data,
    steps_per_epoch=train_data.samples // train_data.batch_size,
    epochs=10,  # Adjust epochs as needed
    validation_data=val_data,
    validation_steps=val_data.samples // val_data.batch_size,
)

# Save the model
model.save("model_grayscale.h5")

# === Step 5: Model Evaluation ===
loss, accuracy = model.evaluate(val_data)
print(f"Validation Loss: {loss}")
print(f"Validation Accuracy: {accuracy}")

# === Step 6: Visualizing Training Results (Optional) ===
import matplotlib.pyplot as plt

# Plot accuracy
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.legend()
plt.title("Training and Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.show()

# Plot loss
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.legend()
plt.title("Training and Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()
