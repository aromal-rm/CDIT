import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os
import matplotlib.pyplot as plt

# -----------------------
# Preprocessing Function
# -----------------------
def preprocess_image(img):
    if len(img.shape) == 3 and img.shape[2] == 3:  # Check if the image is not already grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    denoised = cv2.GaussianBlur(gray, (5, 5), 0)
    resized = cv2.resize(denoised, (128, 128))
    normalized = resized / 255.0
    return np.expand_dims(normalized, axis=0)  # Add channel dimension

# -----------------------
# Custom Dataset
# -----------------------
class CertificateDataset(Dataset):
    def __init__(self, directory):
        self.images = []
        for file in os.listdir(directory):
            img_path = os.path.join(directory, file)
            img = cv2.imread(img_path)
            if img is not None:
                preprocessed_img = preprocess_image(img)
                self.images.append(preprocessed_img)
        self.images = np.array(self.images).astype('float32')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx]

train_dir = 'C:\\Users\\aroma\\Desktop\\cdit'  # Only accepted certificates
dataset = CertificateDataset(train_dir)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# -----------------------
# Build Autoencoder Model
# -----------------------
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, padding=0),  # Adjust padding to 0
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, padding=0)  # Adjust padding to 0
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
autoencoder = Autoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)

# -----------------------
# Train Autoencoder
# -----------------------
num_epochs = 50
for epoch in range(num_epochs):
    for data in dataloader:
        img = data.to(device)
        output = autoencoder(img)
        loss = criterion(output, img)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Save model
torch.save(autoencoder.state_dict(), 'seal_autoencoder.pth')

# -----------------------
# Anomaly Detection
# -----------------------
def calculate_reconstruction_error(img):
    img = preprocess_image(img)
    img = torch.tensor(img).unsqueeze(0).to(device)  # Add batch dimension and move to device
    with torch.no_grad():
        reconstructed = autoencoder(img)
    error = torch.mean((img - reconstructed) ** 2).item()  # Mean squared error
    return error

# Set a threshold based on training images
threshold = np.mean([calculate_reconstruction_error(img) for img in dataset.images]) * 1.5

# Function to predict if a certificate is accepted or not
def predict_acceptance(img_path):
    img = cv2.imread(img_path)
    img = preprocess_image(img)  # Preprocess the image before passing it to the model
    error = calculate_reconstruction_error(img)
    if error < threshold:
        print("Certificate Accepted")
    else:
        print("Certificate Not Accepted")

# Example usage
# img_path = 'path/to/test/image.png'
# predict_acceptance(img_path)
