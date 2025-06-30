import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing import image


PROJECT_DIR = r"C:\Users\omdes\Documents\Om\opencvfun\opencv_album_recog_proj"
MODEL_PATH = os.path.join(PROJECT_DIR, "album_movie_classifier.h5")
LABELS_PATH = os.path.join(PROJECT_DIR, "class_labels.npy")


model = load_model(MODEL_PATH)
class_labels = np.load(LABELS_PATH, allow_pickle=True)


TEST_IMAGE_PATH = os.path.join(PROJECT_DIR, "MoviePosters_raman", "Multi_Label_dataset", "images", "tt0088680.jpg")  # Example: Fight Club

# Load  preprocess image
img = cv2.imread(TEST_IMAGE_PATH)
if img is None:
    print(f" Failed to load image: {TEST_IMAGE_PATH}")
    exit()

img_resized = cv2.resize(img, (64, 64))
img_array = image.img_to_array(img_resized)
img_array = preprocess_input(img_array)
img_array = np.expand_dims(img_array, axis=0)

# Predict
predictions = model.predict(img_array)
predicted_class_index = np.argmax(predictions[0])
confidence = predictions[0][predicted_class_index]

# Show result
print(" Prediction complete!")
print(f" File: {os.path.basename(TEST_IMAGE_PATH)}")
print(f" Predicted Genre: {class_labels[predicted_class_index]}")
print(f" Confidence: {confidence * 100:.2f}%")
