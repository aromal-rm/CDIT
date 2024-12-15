import cv2
import numpy as np
from google.cloud import vision
from google.oauth2 import service_account
import re

# Configure Google Vision API credentials
credentials = service_account.Credentials.from_service_account_file("path/to/your-service-account-key.json")
client = vision.ImageAnnotatorClient(credentials=credentials)

def preprocess_image(image_path):
    # Load the image
    image = cv2.imread(image_path)
    height, width = image.shape[:2]
    
    # Crop the bottom section (assume last 20% of the image height)
    bottom_section = image[int(height * 0.8):, :]
    gray = cv2.cvtColor(bottom_section, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    return bottom_section, edges

def detect_round_seals(edges, original_image):
    # Use Hough Circle Transform
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30, param1=50, param2=30, minRadius=10, maxRadius=100)
    seal_detected = False
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # Draw the detected circle
            cv2.circle(original_image, (i[0], i[1]), i[2], (0, 255, 0), 2)
            cv2.rectangle(original_image, (i[0] - 5, i[1] - 5), (i[0] + 5, i[1] + 5), (0, 128, 255), -1)
            seal_detected = True
    return original_image, seal_detected

def extract_text_with_vision_api(image_path):
    with open(image_path, "rb") as image_file:
        content = image_file.read()
    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    
    # Extract text from the response
    texts = response.text_annotations
    if texts:
        full_text = texts[0].description
        # Extract date using regex
        match = re.search(r'\b\d{2}/\d{2}/\d{4}\b', full_text)
        return full_text, match.group() if match else "No valid date found"
    return "No text detected", "No valid date found"

def process_image(image_path):
    # Preprocess the image to get the bottom section
    bottom_section, edges = preprocess_image(image_path)
    
    # Detect round seals
    processed_image, seal_detected = detect_round_seals(edges, bottom_section)
    
    # Save the cropped bottom section as a temporary file for OCR
    cropped_path = "bottom_section.jpg"
    cv2.imwrite(cropped_path, bottom_section)
    
    # Extract text and date
    full_text, detected_date = extract_text_with_vision_api(cropped_path)
    
    # Display results
    cv2.imshow("Bottom Section with Detected Seal", processed_image)
    print("Seal Detected:", seal_detected)
    print("Extracted Text:", full_text)
    print("Detected Date:", detected_date)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Test the pipeline with the sample certificate
process_image("path/to/sample_certificate.jpg")
