import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import time
import os

# Load model and class labels
PROJECT_DIR = r"C:\Users\omdes\Documents\Om\opencvfun\opencv_album_recog_proj"
MODEL_PATH = os.path.join(PROJECT_DIR, "album_movie_classifier.h5")
LABELS_PATH = os.path.join(PROJECT_DIR, "class_labels.npy")

model = load_model(MODEL_PATH)
class_labels = np.load(LABELS_PATH, allow_pickle=True)

# Initialize camera
cap = cv2.VideoCapture(0)
ret, prev_frame = cap.read()
prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
prev_frame = cv2.GaussianBlur(prev_frame, (21, 21), 0)

print(" Webcam started, press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        print(" Failed to grab frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (21, 21), 0)

    # Calculate difference
    frame_delta = cv2.absdiff(prev_frame, blur)
    thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
    dilated = cv2.dilate(thresh, None, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Get largest moving contour
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) > 5000:  # Adjust threshold as needed
            (x, y, w, h) = cv2.boundingRect(largest)

            roi = frame[y:y+h, x:x+w]
            roi_resized = cv2.resize(roi, (64, 64))
            roi_array = img_to_array(roi_resized) / 255.0
            roi_array = np.expand_dims(roi_array, axis=0)

            preds = model.predict(roi_array, verbose=0)[0]
            top_idx = np.argmax(preds)
            confidence = preds[top_idx] * 100
            label = class_labels[top_idx]

            # Only show predictions if confidence is reasonably high
            if confidence > 10:
                text = f"{label} ({confidence:.1f}%)"
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Show live feed
    cv2.imshow(" Album/Movie Recognizer", frame)

    key = cv2.waitKey(30)
    if key == ord('q'):
        break

    prev_frame = blur

cap.release()
cv2.destroyAllWindows()
