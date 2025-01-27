import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks # type: ignore
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

# Constants
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 30
FOLDS = 5
SEED = 42

# Preprocessing Functions
def preprocess_image(image_path, output_path=None):
    try:
        # Load image in grayscale
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Image at {image_path} could not be loaded.")
        
        # Crop the bottom half of the image
        height, width = img.shape
        bottom_half = img[height // 2:, :]

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(bottom_half, (5, 5), 0)

        # Thresholding to create a binary image
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # If contours are found, correct perspective if needed
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            epsilon = 0.02 * cv2.arcLength(largest_contour, True)
            approx = cv2.approxPolyDP(largest_contour, epsilon, True)

            if len(approx) == 4:  # If a quadrilateral is detected
                # Define points for perspective transform
                pts = approx.reshape(4, 2)
                rect = np.zeros((4, 2), dtype="float32")

                # Sort points to order: top-left, top-right, bottom-right, bottom-left
                s = pts.sum(axis=1)
                rect[0] = pts[np.argmin(s)]
                rect[2] = pts[np.argmax(s)]

                diff = np.diff(pts, axis=1)
                rect[1] = pts[np.argmin(diff)]
                rect[3] = pts[np.argmax(diff)]

                # Calculate the max width and height of the new image
                (tl, tr, br, bl) = rect
                widthA = np.linalg.norm(br - bl)
                widthB = np.linalg.norm(tr - tl)
                maxWidth = max(int(widthA), int(widthB))

                heightA = np.linalg.norm(tr - br)
                heightB = np.linalg.norm(tl - bl)
                maxHeight = max(int(heightA), int(heightB))

                # Perspective transform
                dst = np.array([
                    [0, 0],
                    [maxWidth - 1, 0],
                    [maxWidth - 1, maxHeight - 1],
                    [0, maxHeight - 1]], dtype="float32")

                M = cv2.getPerspectiveTransform(rect, dst)
                warped = cv2.warpPerspective(binary, M, (maxWidth, maxHeight))

                # Resize the perspective-corrected image
                resized_img = cv2.resize(warped, (IMG_SIZE, IMG_SIZE))
            else:
                # If no quadrilateral is detected, use the cropped bottom half
                resized_img = cv2.resize(binary, (IMG_SIZE, IMG_SIZE))
        else:
            # If no contours are found, use the cropped bottom half
            resized_img = cv2.resize(binary, (IMG_SIZE, IMG_SIZE))
        
        # Normalize pixel values
        normalized_img = resized_img / 255.0

        if output_path:
            # Save the preprocessed image
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, (normalized_img * 255).astype(np.uint8))

            # # Display the preprocessed image
            # plt.imshow(normalized_img, cmap='gray')
            # plt.title(f"Preprocessed: {os.path.basename(image_path)}")
            # plt.axis('off')
            # plt.show()

        return normalized_img

    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def preprocess_dataset(input_dir, output_dir=None):
    images, labels = [], []
    phases = ["train", "validation"] if output_dir else ["."]
    base_dir = input_dir

    for phase in phases:
        phase_input_dir = os.path.join(base_dir, phase) if output_dir else base_dir
        phase_output_dir = os.path.join(output_dir, phase) if output_dir else None

        for label, class_name in enumerate(os.listdir(phase_input_dir)):
            class_input_dir = os.path.join(phase_input_dir, class_name)
            class_output_dir = os.path.join(phase_output_dir, class_name) if phase_output_dir else None

            if not os.path.isdir(class_input_dir):
                continue

            for img_name in os.listdir(class_input_dir):
                img_path = os.path.join(class_input_dir, img_name)

                # Skip non-image files
                valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
                if not img_name.lower().endswith(valid_extensions):
                    print(f"Skipping non-image file: {img_name}")
                    continue

                output_path = os.path.join(class_output_dir, img_name) if class_output_dir else None
                preprocessed_img = preprocess_image(img_path, output_path)
                if preprocessed_img is not None:
                    images.append(preprocessed_img)
                    labels.append(label)
                else:
                    print(f"Failed to preprocess: {img_path}")

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
        
        # Plot training history for this fold
        plot_history(history)
        
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
def main(input_dir, output_dir=None):
    images, labels = preprocess_dataset(input_dir, output_dir)
    if images.size == 0 or labels.size == 0:
        print("No valid data found. Exiting.")
        return

    print(f"Preprocessed {len(images)} images.")
    images = images[..., np.newaxis]  # Add channel dimension
    train_and_evaluate(images, labels)

if __name__ == "__main__":
    main("data", "data_preprocessed")
