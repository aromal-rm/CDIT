import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

# Constants
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 20
FOLDS = 5
SEED = 42

# Preprocessing Functions
def preprocess_image(image_path):
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Image at {image_path} could not be loaded.")

        img = cv2.GaussianBlur(img, (5, 5), 0)
        img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            rect = cv2.minAreaRect(largest_contour)
            angle = rect[-1]
            if angle < -45:
                angle += 90
            M = cv2.getRotationMatrix2D((img.shape[1] // 2, img.shape[0] // 2), angle, 1.0)
            img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

        bottom_section = img[img.shape[0] // 2:]
        resized_img = cv2.resize(bottom_section, (IMG_SIZE, IMG_SIZE))
        normalized_img = resized_img / 255.0
        return normalized_img

    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None


def preprocess_directory(input_dir):
    images, labels = [], []
    class_names = os.listdir(input_dir)

    for label, class_name in enumerate(class_names):
        class_path = os.path.join(input_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            preprocessed_img = preprocess_image(img_path)
            if preprocessed_img is not None:
                images.append(preprocessed_img)
                labels.append(label)

    return np.array(images), np.array(labels)

# Model Architecture
def build_model():
    model = models.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)),
        layers.Rescaling(1.0 / 255),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),

        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),

        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),

        layers.Dense(128, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

# Training and Evaluation
def train_and_evaluate(images, labels):
    kf = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(images)):
        print(f"\nStarting Fold {fold + 1}/{FOLDS}...")

        x_train, x_val = images[train_idx], images[val_idx]
        y_train, y_val = labels[train_idx], labels[val_idx]

        model = build_model()

        lr_scheduler = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
        early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        history = model.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[lr_scheduler, early_stopping],
            verbose=1
        )

        val_loss, val_accuracy = model.evaluate(x_val, y_val, verbose=0)
        fold_metrics.append(val_accuracy)

        print(f"Fold {fold + 1} - Validation Accuracy: {val_accuracy:.4f}")
        model.save("try2.h5")

    print("\nCross-validation Accuracy: {:.4f} (+/- {:.4f})".format(np.mean(fold_metrics), np.std(fold_metrics)))

# Visualization Functions
def plot_history(history):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.legend()
    plt.title('Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title('Loss')

    plt.show()

# Main Function
def main(input_dir):
    images, labels = preprocess_directory(input_dir)
    if images.size == 0 or labels.size == 0:
        print("No valid data found. Exiting.")
        return

    images = images[..., np.newaxis]  # Add channel dimension
    train_and_evaluate(images, labels)

if __name__ == "__main__":
    main("/Users/arjun/Documents/CDIT/model/data")
