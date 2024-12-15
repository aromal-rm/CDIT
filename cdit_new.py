import cv2
import numpy as np

def detect_seals(image_path):
    # Step 1: Load the image
    image = cv2.imread(image_path)
    original = image.copy()

    # Step 2: Preprocessing - Resize and Crop Bottom Section
    height, width, _ = image.shape
    # Resize the image for consistency
    scale_percent = 50  # Adjust scale percentage based on image size
    image = cv2.resize(image, (int(width * scale_percent / 100), int(height * scale_percent / 100)))
    height, width, _ = image.shape

    # Focus on the bottom section
    cropped_image = image[int(height * 0.7):, :]

    # Step 3: Correct Skew (Perspective Transformation)
    gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    # Find the largest contour for border detection
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)

        if len(approx) == 4:  # If we detect a quadrilateral
            pts = np.float32([point[0] for point in approx])  # Extract points
            # Ensure consistent order (top-left, top-right, bottom-left, bottom-right)
            rect = np.zeros((4, 2), dtype="float32")
            s = pts.sum(axis=1)
            rect[0] = pts[np.argmin(s)]
            rect[2] = pts[np.argmax(s)]
            diff = np.diff(pts, axis=1)
            rect[1] = pts[np.argmin(diff)]
            rect[3] = pts[np.argmax(diff)]

            # Dimensions of the cropped section
            max_width = width
            max_height = int(height * 0.3)

            dst = np.array([
                [0, 0],
                [max_width - 1, 0],
                [0, max_height - 1],
                [max_width - 1, max_height - 1]
            ], dtype="float32")

            # Perspective transform
            M = cv2.getPerspectiveTransform(rect, dst)
            cropped_image = cv2.warpPerspective(cropped_image, M, (max_width, max_height))

    # Step 4: Continue with Seal Detection
    gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # Morphological closing
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Contour detection
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    seal_candidates = []
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        area = cv2.contourArea(contour)
        if area < 500:  # Minimum area filter
            continue
        circularity = 4 * np.pi * (area / (perimeter * perimeter)) if perimeter > 0 else 0
        if 0.6 <= circularity <= 1.2:  # Circularity check
            seal_candidates.append(contour)

    # Draw detected seals
    result_image = cropped_image.copy()
    for contour in seal_candidates:
        cv2.drawContours(result_image, [contour], -1, (0, 255, 0), 2)

    # Display results
    cv2.imshow('Detected Seals', result_image)
    cv2.imwrite('detected_seals.png', result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
detect_seals('medical_certificate.jpg')
