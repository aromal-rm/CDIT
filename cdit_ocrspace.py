from api import API_KEY
import cv2
import pytesseract
import numpy as np
import re
import os
import requests
from pdf2image import convert_from_path
from PIL import Image

# API configuration for OCR.Space
OCR_SPACE_API_KEY = API_KEY 

# Configure Tesseract for local use
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def convert_image_to_pdf(image_path):
    """
    Convert an image file to a PDF using the img2pdf library.
    """
    pdf_path = image_path.rsplit('.', 1)[0] + '.pdf'
    image = Image.open(image_path)
    image.convert('RGB').save(pdf_path, "PDF")
    return pdf_path

def preprocess_image(image_path):
    """
    Preprocess the image to crop the bottom section and apply grayscale and edge detection.
    """
    # Load the image
    image = cv2.imread(image_path)
    height, width, _ = image.shape
    # Crop the bottom section (adjust proportions based on certificate layout)
    cropped = image[int(height * 0.7):height, 0:width]  # Bottom 30%
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    return cropped, edges

def detect_round_seals(edges, cropped_image):
    """
    Detect circular seals using Hough Circle Transform.
    """
    # Use Hough Circle Transform
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50, param1=50, param2=30, minRadius=10, maxRadius=100)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # Draw the detected circle on the image
            cv2.circle(cropped_image, (i[0], i[1]), i[2], (0, 255, 0), 2)
            cv2.rectangle(cropped_image, (i[0] - 5, i[1] - 5), (i[0] + 5, i[1] + 5), (0, 128, 255), -1)
    return cropped_image, circles

def extract_handwritten_text_with_ocr_space(image_path):
    """
    Extract handwritten text using the OCR.Space API and validate it for date formats.
    """
    # Read the image file
    with open(image_path, 'rb') as f:
        file_data = f.read()

    # Send image to OCR.Space API
    response = requests.post(
        'https://api.ocr.space/parse/image',
        files={'filename': file_data},
        data={
            'apikey': OCR_SPACE_API_KEY,
            'language': 'eng',
            'isOverlayRequired': False
        }
    )

    # Parse the response
    result = response.json()
    if result.get("IsErroredOnProcessing"):
        return "Error in OCR processing", "No valid date found"

    # Extract text
    text = result["ParsedResults"][0]["ParsedText"]
    # Validate date format using regex
    match = re.search(r'\b\d{2}/\d{2}/\d{4}\b', text)
    extracted_date = match.group() if match else "No valid date found"
    return text, extracted_date

def process_pdf_or_image(input_path):
    """
    Main pipeline to process PDFs or images.
    - Converts images to PDF if needed.
    - Extracts pages from PDFs for further processing.
    """
    temp_dir = "temp_pages"
    os.makedirs(temp_dir, exist_ok=True)

    # Check the input type
    if input_path.lower().endswith(('.jpg', '.jpeg', '.png')):
        # Convert image to PDF
        pdf_path = convert_image_to_pdf(input_path)
    elif input_path.lower().endswith('.pdf'):
        pdf_path = input_path
    else:
        raise ValueError("Unsupported file format. Use PDF, JPG, JPEG, or PNG.")

    # Extract images from PDF
    pages = convert_from_path(pdf_path)
    extracted_dates = []

    for idx, page in enumerate(pages):
        page_path = os.path.join(temp_dir, f"page_{idx + 1}.jpg")
        page.save(page_path, "JPEG")
        
        # Preprocess and analyze the page
        cropped_image, edges = preprocess_image(page_path)
        processed_image, circles = detect_round_seals(edges, cropped_image)
        text, extracted_date = extract_handwritten_text_with_ocr_space(page_path)
        extracted_dates.append(extracted_date)
        
        # Display results for each page
        print(f"Page {idx + 1} Extracted Text:")
        print(text)
        print(f"Page {idx + 1} Extracted Date:", extracted_date)
        cv2.imshow(f"Page {idx + 1} - Detected Seals and Text Area", processed_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Cleanup temporary files
    for file in os.listdir(temp_dir):
        os.remove(os.path.join(temp_dir, file))
    os.rmdir(temp_dir)

    return extracted_dates

# Entry point for testing
if __name__ == '__main__':
    # Test with a sample input file (PDF or image)
    test_input_path = "sample_medical_certificate.pdf"  # Replace with your file path
    extracted_dates = process_pdf_or_image(test_input_path)
    print("Extracted Dates from All Pages:", extracted_dates)
