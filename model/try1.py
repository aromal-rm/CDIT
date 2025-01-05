import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import matplotlib.pyplot as plt

# -----------------------
# Preprocessing Function
# -----------------------
def preprocess_image(img):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian Blur for denoising
    denoised = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Threshold to create a binary image
    _, binary = cv2.threshold(denoised, 128, 255, cv2.THRESH_BINARY)
    
    # Find the largest contour for deskewing
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(largest_contour)
        box = cv2.boxPoints(rect)
        box = np.int64(box)
        angle = rect[-1]
        if angle < -45:
            angle += 90
        (h, w) = gray.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        deskewed = cv2.warpAffine(gray, rotation_matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    else:
        deskewed = gray  # Use original if no contours are found

    # Crop to the bottom part of the image (assume bottom 40%)
    h, w = deskewed.shape
    cropped = deskewed[int(h * 0.6):, :]

    # Resize to target size for model input
    resized = cv2.resize(cropped, (128, 128))

    # Normalize pixel values
    normalized = resized / 255.0

    # Return preprocessed image
    return normalized

# -----------------------
# Dataset Preparation
# -----------------------
def preprocess_directory(input_dir, output_dir):
    """Preprocess images in a directory and save to a new directory."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            img_path = os.path.join(root, file)
            img = cv2.imread(img_path)
            preprocessed_img = preprocess_image(img)
            save_path = os.path.join(output_dir, file)
            cv2.imwrite(save_path, (preprocessed_img * 255).astype(np.uint8))

# Preprocess training and validation datasets
preprocess_directory('data/train', 'data/train_preprocessed')
preprocess_directory('data/validation', 'data/validation_preprocessed')

train_dir = 'data/train'
validation_dir = 'data/validation'

# Load the datasets using image_dataset_from_directory
train_data = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=(128, 128),  # Resize all images to 128x128
    batch_size=32,
    label_mode='binary'  # Use binary for two classes
)

validation_data = tf.keras.utils.image_dataset_from_directory(
    validation_dir,
    image_size=(128, 128),  # Resize all images to 128x128
    batch_size=32,
    label_mode='binary'  # Use binary for two classes
)

# Normalize pixel values (0-255) to (0-1)
normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)
train_data = train_data.map(lambda x, y: (normalization_layer(x), y))
validation_data = validation_data.map(lambda x, y: (normalization_layer(x), y))

# Create a simple CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Binary classification
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Print the model summary
model.summary()

# Train the model
history = model.fit(
    train_data,
    validation_data=validation_data,
    epochs=20
)

# Save the trained model
model.save('seal_detection_model.h5')

# -----------------------
# Plot Training and Validation Graphs
# -----------------------
def plot_training_history(history):
    """Plot training and validation accuracy and loss."""
    # Extract accuracy and loss
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    # Create epochs range
    epochs_range = range(len(acc))

    # Plot accuracy
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')

    # Show plots
    plt.show()

# Call the function to plot graphs after training
plot_training_history(history)

# -----------------------
# Testing on New Images
# -----------------------
def predict_acceptance(img_path):
    img = cv2.imread(img_path)
    preprocessed_img = preprocess_image(img)
    preprocessed_img = np.expand_dims(preprocessed_img, axis=-1)  # Add channel dimension
    preprocessed_img = np.expand_dims(preprocessed_img, axis=0)  # Add batch dimension

    prediction = model.predict(preprocessed_img)
    if prediction[0][0] > 0.5:
        print("Certificate Accepted")
    else:
        print("Certificate Not Accepted")

# Test the model
# img_path = 'path/to/test/image.png'
# predict_acceptance(img_path)
