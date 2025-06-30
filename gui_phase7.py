import cv2
import numpy as np
import tensorflow as tf
import webbrowser
import pandas as pd
import os
import time

# CONFIGURATION 
PROJECT_DIR = r"C:\Users\omdes\Documents\Om\opencvfun\opencv_album_recog_proj"
MODEL_PATH = os.path.join(PROJECT_DIR, "album_movie_classifier.h5")
LABELS_PATH = os.path.join(PROJECT_DIR, "train_fixed.csv")
LINKS_CSV_PATH = os.path.join(PROJECT_DIR, "movie_with_links.csv")
CONFIDENCE_THRESHOLD = 0.20
DELAY_SECONDS = 12  # Delay before opening browser

#LOAD MODEL 
model = tf.keras.models.load_model(MODEL_PATH)

# LOAD CLASS LABELS
df_labels = pd.read_csv(LABELS_PATH)
class_labels = df_labels.columns[2:].tolist()  # Assuming first two columns are filename and genre

# LOAD LINKS CSV
df_links = pd.read_csv(LINKS_CSV_PATH)
df_links = df_links.dropna(subset=['genre', 'imdb_link'])  # remove rows with missing genre or link

# Normalize genre field (remove whitespace and sort for consistency)
df_links['genre'] = df_links['genre'].apply(
    lambda g: ", ".join(sorted([s.strip() for s in g.split(",")])) if isinstance(g, str) else ""
)

#INITIALIZE CAMERA 
cap = cv2.VideoCapture(0)
start_time = time.time()
link_opened = False

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    h, w, _ = frame.shape
    box_size = 250
    x1 = w // 2 - box_size // 2
    y1 = h // 2 - box_size // 2
    x2 = x1 + box_size
    y2 = y1 + box_size

    # Draw green style box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Crop, resize, preprocess
    roi = frame[y1:y2, x1:x2]
    if roi.shape[0] > 0 and roi.shape[1] > 0:
        img = cv2.resize(roi, (64, 64))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array / 255.0, axis=0)

        predictions = model.predict(img_array)[0]
        max_index = np.argmax(predictions)
        max_conf = predictions[max_index]
        predicted_class = class_labels[max_index]

        # Display prediction on screen
        label_text = f"{predicted_class} ({max_conf * 100:.1f}%)"
        if max_conf > CONFIDENCE_THRESHOLD:
            cv2.putText(frame, label_text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # Wait for delay before opening link
            if not link_opened and (time.time() - start_time) > DELAY_SECONDS:
                # Normalize predicted genre same way as in df_links
                predicted_class_norm = ", ".join(sorted([s.strip() for s in predicted_class.split(",")]))
                matched_row = df_links[df_links['genre'] == predicted_class_norm]

                if not matched_row.empty:
                    link = matched_row.iloc[0]['imdb_link']
                    webbrowser.open(link)
                    print(f" Opening: {link}")
                    link_opened = True
                else:
                    print(" No matching link found for:", predicted_class)
        else:
            link_opened = False  # Reset if confidence drops

    cv2.imshow("Movie Recognizer", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
