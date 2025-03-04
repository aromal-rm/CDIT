import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

# Constants
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 30
FOLDS = 5
SEED = 42

def preprocess_image(image_path, output_path=None):
    """
    Preprocess a certificate image by:
      - Loading the full image in grayscale.
      - Applying Gaussian blur and OTSU thresholding.
      - Using morphological closing to reduce noise.
      - Detecting contours and, if a significant quadrilateral is found,
        applying a perspective transform for correction.
      - Resizing to IMG_SIZE x IMG_SIZE and normalizing pixel values.
    
    Args:
        image_path (str): Path to the certificate image.
        output_path (str, optional): If provided, the preprocessed image is saved.
    
    Returns:
        np.array: The preprocessed image or None on failure.
    """
    try:
        # Load the full image in grayscale
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Image at {image_path} could not be loaded.")
            
        orig_area = img.shape[0] * img.shape[1]
        
        # Enhance image: apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(img, (5, 5), 0)
        
        # OTSU thresholding to obtain a binary image
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Apply morphological closing to fill small gaps
        kernel = np.ones((5, 5), np.uint8)
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Find contours on the processed image
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Choose the largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            # Proceed only if the contour area is significant (likely the certificate)
            if cv2.contourArea(largest_contour) > 0.2 * orig_area:
                epsilon = 0.02 * cv2.arcLength(largest_contour, True)
                approx = cv2.approxPolyDP(largest_contour, epsilon, True)
                if len(approx) == 4:
                    # Order the points: top-left, top-right, bottom-right, bottom-left
                    pts = approx.reshape(4, 2)
                    rect = np.zeros((4, 2), dtype="float32")
                    s = pts.sum(axis=1)
                    rect[0] = pts[np.argmin(s)]
                    rect[2] = pts[np.argmax(s)]
                    diff = np.diff(pts, axis=1)
                    rect[1] = pts[np.argmin(diff)]
                    rect[3] = pts[np.argmax(diff)]
                    
                    # Compute new dimensions based on the ordered points
                    (tl, tr, br, bl) = rect
                    widthA = np.linalg.norm(br - bl)
                    widthB = np.linalg.norm(tr - tl)
                    maxWidth = max(int(widthA), int(widthB))
                    
                    heightA = np.linalg.norm(tr - br)
                    heightB = np.linalg.norm(tl - bl)
                    maxHeight = max(int(heightA), int(heightB))
                    
                    # Define destination points for the perspective transform
                    dst = np.array([
                        [0, 0],
                        [maxWidth - 1, 0],
                        [maxWidth - 1, maxHeight - 1],
                        [0, maxHeight - 1]], dtype="float32")
                    
                    M = cv2.getPerspectiveTransform(rect, dst)
                    warped = cv2.warpPerspective(binary, M, (maxWidth, maxHeight))
                    preprocessed_img = cv2.resize(warped, (IMG_SIZE, IMG_SIZE))
                else:
                    # No clear quadrilateral detected; fallback to resizing binary image.
                    preprocessed_img = cv2.resize(binary, (IMG_SIZE, IMG_SIZE))
            else:
                # Contour area too small; fallback to resizing the binary image.
                preprocessed_img = cv2.resize(binary, (IMG_SIZE, IMG_SIZE))
        else:
            # No contours found; fallback to resized binary image.
            preprocessed_img = cv2.resize(binary, (IMG_SIZE, IMG_SIZE))
        
        # Normalize the image to [0,1]
        normalized_img = preprocessed_img.astype('float32') / 255.0
        
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, (normalized_img * 255).astype(np.uint8))
            
        return normalized_img
    
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def preprocess_dataset(input_dir, output_dir=None):
    """
    Preprocess all certificate images within the dataset.
    
    The dataset is expected to be organized in class subdirectories.
    
    Args:
        input_dir (str): Directory with raw certificate images.
        output_dir (str, optional): If provided, saves the preprocessed images.
    
    Returns:
        tuple: (np.array of images, np.array of labels)
    """
    images, labels = []
    labels = []
    phases = ["train", "validation"] if output_dir else ["."]
    base_dir = input_dir
    
    for phase in phases:
        phase_input_dir = os.path.join(base_dir, phase) if output_dir else base_dir
        phase_output_dir = os.path.join(output_dir, phase) if output_dir else None
        
        # Sorting ensures consistent label assignment
        for label, class_name in enumerate(sorted(os.listdir(phase_input_dir))):
            class_input_dir = os.path.join(phase_input_dir, class_name)
            if not os.path.isdir(class_input_dir):
                continue
            
            class_output_dir = os.path.join(phase_output_dir, class_name) if phase_output_dir else None
            
            for img_name in os.listdir(class_input_dir):
                img_path = os.path.join(class_input_dir, img_name)
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

def build_model():
    """
    Build and compile the CNN model.
    
    The model is designed for binary classification (accepting vs. not accepting certificates).
    
    Returns:
        tf.keras.Model: Compiled model.
    """
    model = models.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)),
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

def plot_history(history):
    """
    Plot the training history (accuracy and loss curves).
    
    Args:
        history: History object returned by model.fit.
    """
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def train_and_evaluate(images, labels):
    """
    Train and evaluate the model using K-Fold cross-validation.
    
    Args:
        images (np.array): Preprocessed certificate images.
        labels (np.array): Corresponding class labels.
    """
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
        plot_history(history)
    
    model.save("certificate_classifier.h5")
    print("\nCross-validation Accuracy: {:.4f} (+/- {:.4f})".format(np.mean(fold_metrics), np.std(fold_metrics)))

def main(input_dir, output_dir=None):
    """
    Main function to preprocess the dataset and train the classifier.
    
    Args:
        input_dir (str): Directory containing raw certificate images.
        output_dir (str, optional): Directory to save preprocessed images.
    """
    images, labels = preprocess_dataset(input_dir, output_dir)
    if images.size == 0 or labels.size == 0:
        print("No valid data found. Exiting.")
        return
    print(f"Preprocessed {len(images)} images.")
    images = images[..., np.newaxis]  # Add channel dimension for grayscale
    train_and_evaluate(images, labels)

if __name__ == "__main__":
    # Adjust the directories as needed.
    main("data", "data_preprocessed")
