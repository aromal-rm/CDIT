import cv2
import numpy as np
import pytesseract
import torch
from torch import nn
from PIL import Image
import os

# --------------------------
# 1. Unified Preprocessing
# --------------------------

def preprocess_certificate(img_path):
    # Load image
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Adaptive thresholding
    processed = cv2.adaptiveThreshold(gray, 255, 
                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 11, 2)
    
    # Deskew using text orientation
    coords = np.column_stack(np.where(processed > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    return rotated

# --------------------------
# 2. Format Verification
# --------------------------

def verify_format(img):
    # OCR with confidence scores
    custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.:,- "'
    data = pytesseract.image_to_data(img, config=custom_config, output_type=pytesseract.Output.DICT)
    
    required_fields = {
        'MEDICAL CERTIFICATE': 0.7,
        'Name and address of the Patient': 0.7,
        'Signature': 0.65,
        'Head of the Hospitals': 0.65
    }
    
    found_fields = {}
    for i, text in enumerate(data['text']):
        conf = float(data['conf'][i])/100
        text = text.strip()
        for field in required_fields:
            if text.lower() in field.lower() and conf >= required_fields[field]:
                found_fields[field] = True
    
    return len(found_fields) == len(required_fields)

# --------------------------
# 3. Seal Detection CNN
# --------------------------

class SealDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*16*16, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.layers(x)

def detect_seal(img, model):
    # Sliding window approach
    seal_regions = []
    scale = 1.0
    for _ in range(3):
        scaled_img = cv2.resize(img, (0,0), fx=scale, fy=scale)
        for y in range(0, scaled_img.shape[0] - 64, 32):
            for x in range(0, scaled_img.shape[1] - 64, 32):
                patch = scaled_img[y:y+64, x:x+64]
                tensor = torch.tensor(patch/255.0).permute(2,0,1).float().unsqueeze(0)
                with torch.no_grad():
                    prob = model(tensor).item()
                if prob > 0.85:
                    seal_regions.append((x, y, 64, 64))
        scale *= 0.7
    
    return len(seal_regions) > 0

# --------------------------
# 4. Signature Verification
# --------------------------

def verify_signature(img):
    # Focus on bottom-right quadrant (common signature location)
    h, w = img.shape[:2]
    roi = img[int(h*0.7):h-20, int(w*0.6):w-20]
    
    # Edge density analysis
    edges = cv2.Canny(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), 50, 150)
    edge_pixels = np.count_nonzero(edges)
    
    # Texture analysis
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    return edge_pixels > 1000 and laplacian > 500

# --------------------------
# 5. Integrated Verification
# --------------------------

class CertificateValidator:
    def __init__(self):
        self.seal_model = self.load_seal_model()
        self.seal_model.eval()
        
    def load_seal_model(self):
        model = SealDetector()
        # Load pretrained weights or train with synthetic data
        if os.path.exists('seal_detector.pth'):
            model.load_state_dict(torch.load('seal_detector.pth'))
        return model
    
    def validate(self, img_path):
        try:
            # Preprocess
            img = preprocess_certificate(img_path)
            
            # Format check
            if not verify_format(img):
                return False, "Invalid format"
            
            # Seal check
            if not detect_seal(img, self.seal_model):
                return False, "No seal detected"
            
            # Signature check
            if not verify_signature(img):
                return False, "Invalid signature"
            
            return True, "Valid certificate"
        
        except Exception as e:
            return False, f"Validation error: {str(e)}"

# --------------------------
# 6. Training with Weak Supervision
# --------------------------

def train_seal_detector(certificate_dir):
    # Generate synthetic seals on certificates
    model = SealDetector()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Weak supervision: Assume 50% of certificates have valid seals
    for epoch in range(10):
        for cert_path in os.listdir(certificate_dir):
            img = preprocess_certificate(os.path.join(certificate_dir, cert_path))
            
            # Generate random synthetic seal
            if np.random.rand() > 0.5:
                seal = generate_synthetic_seal()
                x = np.random.randint(0, img.shape[1]-64)
                y = np.random.randint(0, img.shape[0]-64)
                img[y:y+64, x:x+64] = seal
                label = 1
            else:
                label = 0
            
            # Convert to tensor
            tensor = torch.tensor(img/255.0).permute(2,0,1).float().unsqueeze(0)
            
            # Training step
            optimizer.zero_grad()
            output = model(tensor)
            loss = nn.BCELoss()(output, torch.tensor([[label]]).float())
            loss.backward()
            optimizer.step()
        
        torch.save(model.state_dict(), 'seal_detector.pth')

def generate_synthetic_seal():
    # Create synthetic seal pattern
    seal = np.zeros((64, 64, 3), dtype=np.uint8)
    color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
    cv2.circle(seal, (32, 32), 30, color, 2)
    cv2.putText(seal, "SEAL", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return seal

# --------------------------
# Usage
# --------------------------

if __name__ == "__main__":
    # First train with your certificates
    train_seal_detector('path/to/medical_certificates')
    
    # Then validate certificates
    validator = CertificateValidator()
    result, message = validator.validate('path/to/certificate.jpg')
    print(f"Result: {result}, Message: {message}")