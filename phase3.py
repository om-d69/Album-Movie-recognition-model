import os
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing import image


PROJECT_DIR = "C:\\Users\\omdes\\Documents\\Om\\opencvfun\\opencv_album_recog_proj"


MOVIE_DATASET_PATH = os.path.join(PROJECT_DIR, "MoviePosters_raman", "Multi_Label_dataset", "images")
MOVIE_CSV_PATH = os.path.join(PROJECT_DIR, "MoviePosters_raman", "Multi_Label_dataset", "train.csv")


def read_csv_with_fallback_encoding(file_path, encodings=['utf-8', 'latin1', 'cp1252', 'ISO-8859-1']):
    for encoding in encodings:
        try:
            print(f"Trying to read {file_path} with {encoding} encoding...")
            return pd.read_csv(file_path, encoding=encoding)
        except UnicodeDecodeError:
            print(f"Failed with {encoding} encoding, trying next...")

    print("All standard encodings failed. Using utf-8 with errors='replace'...")
    return pd.read_csv(file_path, encoding='utf-8', errors='replace')


print("🧹 Cleaning up old movie batch files...")
for file in os.listdir(PROJECT_DIR):
    if file.startswith("images_batch_movie") or file.startswith("labels_batch_movie"):
        os.remove(os.path.join(PROJECT_DIR, file))
        print(f"🗑️ Deleted {file}")


print("📥 Loading movie CSV...")
movie_labels = read_csv_with_fallback_encoding(MOVIE_CSV_PATH)


if not {'filename', 'genre'}.issubset(movie_labels.columns):
    raise ValueError("CSV must contain 'filename' and 'genre' columns.")


movie_labels = movie_labels[movie_labels['filename'].notna()]


movie_labels.to_csv(os.path.join(PROJECT_DIR, "MoviePosters_Raman_fixed.csv"), index=False)


def save_batches(folder_path, labels_df, label_key, prefix, batch_size=5000):
    images, labels = [], []
    batch_count = 0
    processed = 0

    for idx, row in labels_df.iterrows():
        filename = str(row['filename']).strip()
        label = row[label_key]
        img_path = os.path.join(folder_path, filename)

        if not os.path.exists(img_path):
            print(f"❌ Image not found: {img_path}")
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️ Failed to read image: {img_path}")
            continue

        img = cv2.resize(img, (64, 64))
        img = img.astype(np.uint8)

        images.append(img)
        labels.append(label)
        processed += 1

        if len(images) == batch_size:
            np.save(os.path.join(PROJECT_DIR, f"images_batch_{prefix}_{batch_count}.npy"), np.array(images, dtype=np.uint8))
            np.save(os.path.join(PROJECT_DIR, f"labels_batch_{prefix}_{batch_count}.npy"), np.array(labels))
            print(f"💾 Saved batch {batch_count} with {len(images)} images")
            images, labels = [], []
            batch_count += 1

    if images:
        np.save(os.path.join(PROJECT_DIR, f"images_batch_{prefix}_{batch_count}.npy"), np.array(images, dtype=np.uint8))
        np.save(os.path.join(PROJECT_DIR, f"labels_batch_{prefix}_{batch_count}.npy"), np.array(labels))
        print(f"💾 Saved final batch {batch_count} with {len(images)} images")

    print(f"✅ Done processing {processed} images from {prefix} dataset.")

# Save only movie batches
print("📦 Processing movie dataset...")
save_batches(MOVIE_DATASET_PATH, movie_labels, 'genre', 'movie')

