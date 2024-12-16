import cv2
import numpy as np

def crop_bottom_section(image, crop_ratio=0.4):
    """Crop the bottom section of the image."""
    h, w = image.shape[:2]
    start_y = int(h * (1 - crop_ratio))
    cropped_image = image[start_y:h, :]
    return cropped_image

def correct_skew(image):
    """Correct the skew of the image using Hough lines."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    
    if lines is not None:
        angles = []
        for rho, theta in lines[:, 0]:
            angle = (theta * 180 / np.pi) - 90  # Convert to degrees
            angles.append(angle)
        median_angle = np.median(angles)

        # Correct skew by rotating the image
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        corrected_image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return corrected_image
    else:
        print("[INFO] No skew detected.")
        return image

def enhance_image(image):
    """Convert to grayscale and enhance the image contrast."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    enhanced = cv2.equalizeHist(gray)  # Histogram equalization for contrast enhancement
    return enhanced

def apply_morphology(image):
    """Apply morphological operations to reduce noise."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    morphed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    return morphed

def detect_edges(image):
    """Apply Gaussian blur and detect edges using Canny."""
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    return edges

def filter_circles(circles, height_threshold, min_radius=10, max_radius=100):
    """Filter circles based on their position and radius."""
    filtered = []
    for circle in circles[0, :]:
        x, y, r = circle
        if y > height_threshold / 2 and min_radius < r < max_radius:  # Filter based on y-position and size
            filtered.append((x, y, r))
    return filtered

def detect_seals(image_path, output_path="final_seal_detection_result.png"):
    """Main function to detect seals in an image."""
    print("[INFO] Loading image...")
    image = cv2.imread(image_path)
    if image is None:
        print("[ERROR] Image could not be loaded.")
        return
    
    # Crop bottom section
    print("[INFO] Cropping bottom section...")
    cropped_image = crop_bottom_section(image)

    # Correct skew
    print("[INFO] Correcting skew...")
    corrected_image = correct_skew(cropped_image)

    # Enhance contrast
    print("[INFO] Enhancing contrast...")
    enhanced_image = enhance_image(corrected_image)

    # Reduce noise using morphology
    print("[INFO] Applying morphological operations...")
    morphed_image = apply_morphology(enhanced_image)

    # Detect edges
    print("[INFO] Detecting edges...")
    edges = detect_edges(morphed_image)

    # Detect circles using Hough Circle Transform
    print("[INFO] Applying Hough Circle Transform...")
    circles = cv2.HoughCircles(morphed_image, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
                               param1=65, param2=45, minRadius=50, maxRadius=90)
    
    result_image = corrected_image.copy()

    if circles is not None:
        filtered_circles = filter_circles(circles, corrected_image.shape[0])
        print(f"[INFO] Detected {len(filtered_circles)} potential seals after filtering.")
        for x, y, r in filtered_circles:
            cv2.circle(result_image, (int(x), int(y)), int(r), (0, 255, 0), 2)  # Seal boundary
            cv2.circle(result_image, (int(x), int(y)), 2, (0, 0, 255), 3)      # Center point
    else:
        print("[INFO] No circles detected.")
    
    # Save the result
    print("[INFO] Saving final result...")
    cv2.imwrite(output_path, result_image)
    print(f"[INFO] Final result saved to: {output_path}")

# Entry point
if __name__ == "__main__":
    image_path = "med_cert_5.jpg"  # Replace with your input image path
    detect_seals(image_path)
