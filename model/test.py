import os
from keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# --- Step 1: Load the Model ---
model = load_model('try2.h5')
print("Model loaded successfully.")

# --- Step 2: Helper Functions ---

# Preprocess a single image
def preprocess_image(image_path, target_size=(128, 128)):
    """Load and preprocess a single image."""
    img = load_img(image_path, target_size=target_size)  # Resize the image
    img_array = img_to_array(img)  # Convert to NumPy array
    img_array = img_array / 255.0  # Rescale pixel values to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Make prediction for a single image
def predict_single_image(image_path):
    """Predict the class of a single image."""
    img_array = preprocess_image(image_path)
    prediction = model.predict(img_array)
    return prediction[0][0]

# Predict for a batch of images in a directory
def predict_images_in_directory(directory_path):
    """Predict classes for all images in a directory."""
    results = {}
    for image_name in os.listdir(directory_path):
        image_path = os.path.join(directory_path, image_name)
        if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):  # Check valid image extensions
            prediction = predict_single_image(image_path)
            results[image_name] = "Class 1 (positive)" if prediction > 0.5 else "Class 0 (negative)"
    return results

# --- Step 3: Test the Model ---

# Option 1: Test a single image
single_image_path = '/Users/arjun/Documents/CDIT/confidential/med_cert_6.png'  # Replace with your image path
single_prediction = predict_single_image(single_image_path)
if single_prediction > 0.5:
    print(f"The image '{os.path.basename(single_image_path)}' belongs to Class 1 (positive).")
else:
    print(f"The image '{os.path.basename(single_image_path)}' belongs to Class 0 (negative).")

# # Option 2: Test multiple images in a directory
# directory_path = 'path_to_your_image_directory'  # Replace with your directory path
# batch_predictions = predict_images_in_directory(directory_path)

# print("\nBatch Predictions:")
# for image_name, prediction in batch_predictions.items():
#     print(f"{image_name}: {prediction}")
