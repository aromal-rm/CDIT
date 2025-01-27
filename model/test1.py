import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Load the saved model
model = load_model("try2.h5")

# Define image preprocessing parameters
img_height = 150
img_width = 150

# Function to preprocess a single image
def preprocess_image(image_path):
    # Load the image
    img = load_img(image_path, target_size=(img_height, img_width))
    # Convert image to array
    img_array = img_to_array(img)
    # Rescale pixel values to [0, 1]
    img_array = img_array / 255.0
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Predict for a single image
def predict_single_image(image_path, model):
    # Preprocess the image
    preprocessed_image = preprocess_image(image_path)
    # Make prediction
    prediction = model.predict(preprocessed_image, verbose=0)[0][0]
    # Apply threshold for binary classification
    predicted_class_label = "class_1" if prediction >= 0.5 else "class_0"
    return predicted_class_label

# Example usage
image_path = "/Users/arjun/Documents/CDIT/confidential/med_cert_6.png"  # Replace with the path to your image
try:
    predicted_label = predict_single_image(image_path, model)
    print(f"Predicted label: {predicted_label}")
except Exception as e:
    print(f"Error during prediction: {e}")
