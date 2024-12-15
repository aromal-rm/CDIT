import cv2
import numpy as np
import pytesseract
import re
import os

# Configure Tesseract path (ensure Tesseract is installed and configured properly)
TESSERACT_PATH = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
if not os.path.exists(TESSERACT_PATH):
    raise FileNotFoundError("Tesseract executable not found. Check the path!")
pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

def preprocess_image(image_path, crop_height_ratio=0.6):
    """
    Preprocess the image to extract the bottom section and apply thresholding.
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Error: Image at '{image_path}' not found.")
    height, width, _ = image.shape

    # Crop the bottom portion based on ratio
    cropped = image[int(height * crop_height_ratio):, :]

    # Convert to grayscale and apply Gaussian Blur
    gray = cv2.GaussianBlur(cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY), (5, 5), 0)

    # Adaptive Thresholding for better handling of varying lighting conditions
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    return cropped, binary

def detect_round_seals(binary, original_image, min_area=500, circularity_range=(0.7, 1.2)):
    """
    Detect round seals using contours and morphological operations.
    """
    # Morphological operations to close gaps in the binary image
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    morphed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detected_seals = []
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue  # Avoid division by zero
        
        # Calculate circularity
        circularity = (4 * np.pi * area) / (perimeter ** 2)
        if circularity_range[0] < circularity <= circularity_range[1] and area > min_area:
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)
            
            # Additional check for the KIMSHEALTH seal
            seal_roi = original_image[center[1] - radius:center[1] + radius, center[0] - radius:center[0] + radius]
            seal_gray = cv2.cvtColor(seal_roi, cv2.COLOR_BGR2GRAY)
            _, seal_binary = cv2.threshold(seal_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            if np.mean(seal_binary) > 128:
                cv2.circle(original_image, center, radius, (0, 255, 0), 2)  # Draw seal outline
                detected_seals.append((center, radius))

    return original_image, detected_seals

def extract_text(image, config='--psm 6'):
    """
    Extract text from the image using Tesseract OCR.
    """
    text = pytesseract.image_to_string(image, config=config)
    return text.strip()

def extract_date_from_text(text):
    """
    Extract a date in dd/mm/yyyy format from the given text using regex.
    """
    match = re.search(r'\b\d{2}/\d{2}/\d{4}\b', text)
    return match.group() if match else "No valid date found"

def main(image_path):
    """
    Main function to preprocess the image, detect seals, and extract handwritten text.
    """
    try:
        # Preprocess image
        cropped_image, binary_image = preprocess_image(image_path)

        # Detect round seals
        processed_image, seals = detect_round_seals(binary_image, cropped_image)

        # Extract text and validate dates
        extracted_text = extract_text(cropped_image)
        extracted_date = extract_date_from_text(extracted_text)

        # Results
        print("Extracted Text:")
        print(extracted_text)
        print("Extracted Date:", extracted_date)
        print(f"Number of detected seals: {len(seals)}")

        # Display the processed images
        cv2.imshow("Original Image with Detected Seals", processed_image)
        cv2.imshow("Binary Image", binary_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Optionally save the processed image
        cv2.imwrite("processed_image_with_seals.jpg", processed_image)

    except Exception as e:
        print(f"Error: {e}")

# Entry point
if __name__ == "__main__":
    test_image_path = "med_cert.jpg"  # Replace with your image path
    main(test_image_path)