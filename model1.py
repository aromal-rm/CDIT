import cv2
import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# -----------------------
# STEP 1: Preprocessing
# -----------------------
def preprocess_image(img_path):
    """Preprocess the image to extract contours."""
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)  # Remove noise with Gaussian blur
    edges = cv2.Canny(blurred, 50, 150)  # Edge detection using Canny

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output = img.copy()
    for contour in contours:
        # Approximate circular contours
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        area = cv2.contourArea(contour)
        if len(approx) > 8 and area > 100:  # Filter for circular shapes
            cv2.drawContours(output, [contour], -1, (0, 255, 0), 2)
    return output

# Test preprocessing on an example
# img_path = 'image.png'  # Change to your image path
# preprocessed_img = preprocess_image(img_path)
# plt.imshow(cv2.cvtColor(preprocessed_img, cv2.COLOR_BGR2RGB))
# plt.title("Preprocessed Image - Contours Detected")
# plt.show()

# -----------------------
# STEP 2: Dataset Preparation
# -----------------------
# Assuming images are organized as:
# data/
#   train/
#       seal/ (Images with seals)
#       no_seal/ (Images without seals)
#   validation/
#       seal/
#       no_seal/

data_dir = 'data/train'  # Modify as per your directory structure

# Data augmentation for better training
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=10, zoom_range=0.2,
                                   width_shift_range=0.1, height_shift_range=0.1)
train_data = train_datagen.flow_from_directory(data_dir, target_size=(128, 128),
                                               batch_size=32, class_mode='binary')

validation_datagen = ImageDataGenerator(rescale=1./255)
validation_data = validation_datagen.flow_from_directory('data/validation',
                                                         target_size=(128, 128),
                                                         batch_size=32, class_mode='binary')

# -----------------------
# STEP 3: Model Creation
# -----------------------
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),  # Prevent overfitting
    Dense(1, activation='sigmoid')  # Binary classification
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# -----------------------
# STEP 4: Training the Model
# -----------------------
history = model.fit(train_data, validation_data=validation_data, epochs=10)

# -----------------------
# STEP 5: Evaluating the Model
# -----------------------
loss, acc = model.evaluate(validation_data)
print(f"Validation Accuracy: {acc:.2f}")

# -----------------------
# STEP 6: Testing on New Images
# -----------------------
def predict_seal(img_path):
    img = cv2.imread(img_path)
    img_resized = cv2.resize(img, (128, 128))
    img_resized = img_resized / 255.0  # Normalize
    img_resized = np.expand_dims(img_resized, axis=0)

    prediction = model.predict(img_resized)
    if prediction[0][0] > 0.5:
        print("Seal Detected")
    else:
        print("No Seal Detected")

# Test prediction
# predict_seal(img_path)
