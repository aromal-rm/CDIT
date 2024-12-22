import cv2
import numpy as np

def detect_seals(image_path):
    # Step 1: Load the image
    print("[INFO] Loading image...")
    image = cv2.imread(image_path)
    if image is None:
        print("[ERROR] Unable to load image.")
        return
    original = image.copy()

    # Step 2: Preprocessing - Resize and Crop Bottom Section
    print("[INFO] Resizing and cropping bottom section...")
    height, width, _ = image.shape
    scale_percent = 50  # Adjust scale percentage based on image size
    image = cv2.resize(image, (int(width * scale_percent / 100), int(height * scale_percent / 100)))
    height, width, _ = image.shape

    # Focus on the bottom section
    cropped_image = image[int(height * 0.6):, :]
    print(f"[DEBUG] Cropped image size: {cropped_image.shape}")

    # Step 3: Preprocess for Skewed Images
    print("[INFO] Preprocessing image for contour detection...")
    gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # Morphological closing
    print("[INFO] Applying morphological operations...")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Step 4: Find Contours
    print("[INFO] Detecting contours...")
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"[DEBUG] Total contours found: {len(contours)}")

    # Step 5: Filter Based on Area and Circularity
    seal_candidates = []
    min_area = (height * width) * 0.005  # Minimum area as a percentage of cropped size
    max_area = (height * width) * 0.5   # Maximum area to exclude very large objects
    print(f"[DEBUG] Area thresholds: min_area={min_area:.2f}, max_area={max_area:.2f}")

    for i, contour in enumerate(contours):
        perimeter = cv2.arcLength(contour, True)
        area = cv2.contourArea(contour)
        print(f"[DEBUG] Contour {i}: Area={area:.2f}, Perimeter={perimeter:.2f}")
        
        if area < min_area or area > max_area:  # Filter by area
            print(f"[INFO] Contour {i} rejected (area outside threshold).")
            continue

        # Circularity Check
        circularity = 4 * np.pi * (area / (perimeter * perimeter)) if perimeter > 0 else 0
        print(f"[DEBUG] Contour {i}: Circularity={circularity:.2f}")
        if 0.6 <= circularity <= 1.2:  # Accept imperfect circular shapes
            print(f"[INFO] Contour {i} accepted as a seal candidate.")
            seal_candidates.append(contour)
        else:
            print(f"[INFO] Contour {i} rejected (circularity outside threshold).")

    print(f"[DEBUG] Total valid seal candidates: {len(seal_candidates)}")

    # Step 6: Draw Detected Seals
    print("[INFO] Drawing detected seals...")
    result_image = cropped_image.copy()
    for contour in seal_candidates:
        cv2.drawContours(result_image, [contour], -1, (0, 255, 0), 2)

    # Display results
    print("[INFO] Displaying and saving result image...")
    cv2.imshow('Detected Seals', result_image)
    cv2.imwrite('detected_seals_filtered_debug.png', result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


detect_seals('med_cert_6.png')
