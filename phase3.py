import cv2
import numpy as np
import tensorflow as tf
import webbrowser
import pandas as pd
import os

#  CONFIGURATION 
PROJECT_DIR = r"C:\Users\omdes\Documents\Om\opencvfun\opencv_album_recog_proj"
MODEL_PATH = os.path.join(PROJECT_DIR, "album_movie_classifier.h5")
LABELS_PATH = os.path.join(PROJECT_DIR, "train_fixed.csv")  # just for class labels
LINKS_CSV_PATH = os.path.join(PROJECT_DIR, "movie_with_links.csv")
CONFIDENCE_THRESHOLD = 0.20

# LOAD DATA 
model = tf.keras.models.load_model(MODEL_PATH)
df_links = pd.read_csv(LINKS_CSV_PATH)
class_labels = pd.read_csv(LABELS_PATH).columns[2:]  # Skip filename & genre

# VIDEO CAPTURE 
cap = cv2.VideoCapture(0)
link_opened = False  # Ensure link opens only once per detection

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Define scanning area like FaceID
    h, w, _ = frame.shape
    box_size = 250
    x1 = w // 2 - box_size // 2
    y1 = h // 2 - box_size // 2
    x2 = x1 + box_size
    y2 = y1 + box_size

    # Draw green detection box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Crop and preprocess
    roi = frame[y1:y2, x1:x2]
    if roi.shape[0] != 0 and roi.shape[1] != 0:
        img = cv2.resize(roi, (64, 64))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array / 255.0, axis=0)

        predictions = model.predict(img_array)[0]
        max_index = np.argmax(predictions)
        max_conf = predictions[max_index]
        predicted_class = class_labels[max_index]

        # Show prediction only if confidence is high enough
        if max_conf > CONFIDENCE_THRESHOLD:
            label_text = f"{predicted_class} ({max_conf*100:.1f}%)"
            cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # Open link only once
            if not link_opened:
                # Try to match genre to a row and get corresponding IMDb link
                match_row = df_links[df_links['genre'] == predicted_class]
                if not match_row.empty:
                    imdb_link = match_row.iloc[0]['imdb_link']
                    webbrowser.open(imdb_link)
                    link_opened = True
        else:
            link_opened = False  # Reset if low confidence
    else:
        link_opened = False

    cv2.imshow("Album/Movie Recognizer", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


