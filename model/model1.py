import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# -----------------------
# Dataset Preparation
# -----------------------
data_dir = 'data/train'  # Path to training data
validation_dir = 'data/validation'  # Path to validation data

train_datagen = ImageDataGenerator(
    rescale=1./255,         
    rotation_range=10,      
    zoom_range=0.2,         
    width_shift_range=0.1,  
    height_shift_range=0.1  
)
train_data = train_datagen.flow_from_directory(
    data_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary'
)

validation_datagen = ImageDataGenerator(rescale=1./255)
validation_data = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary'
)

# -----------------------
# Model Creation
# -----------------------
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# -----------------------
# Training the Model
# -----------------------
history = model.fit(train_data, validation_data=validation_data, epochs=20)
model.save('CDIT.h5')

# -----------------------
# # Testing on New Images (Commented)
# -----------------------
# def predict_acceptance(img_path):
#     img = cv2.imread(img_path)
#     img_resized = cv2.resize(img, (128, 128))
#     img_resized = img_resized / 255.0
#     img_resized = np.expand_dims(img_resized, axis=0)

#     prediction = model.predict(img_resized)
#     if prediction[0][0] > 0.5:
#         print("Certificate Accepted")
#     else:
#         print("Certificate Not Accepted")

# Test the model
# img_path = 'path/to/test/image.png'
# predict_acceptance(img_path)
