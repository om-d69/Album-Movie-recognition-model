import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam

PROJECT_DIR = r"C:\Users\omdes\Documents\Om\opencvfun\opencv_album_recog_proj"

def load_batches(prefix, max_batches=10):
    images, labels = [], []
    for i in range(max_batches):
        img_path = os.path.join(PROJECT_DIR, f"images_batch_{prefix}_{i}.npy")
        lbl_path = os.path.join(PROJECT_DIR, f"labels_batch_{prefix}_{i}.npy")

        if not os.path.exists(img_path) or not os.path.exists(lbl_path):
            print(f"!!Batch{i} missing. Skipping")
            continue

        images.append(np.load(img_path))
        labels.append(np.load(lbl_path, allow_pickle=True))

    if not images:
        raise RuntimeError("No batches found. Run Phase 3 again.")

    X = np.concatenate(images, axis=0)
    y = np.concatenate(labels, axis=0)
    return X, y

X, y = load_batches("movie", max_batches=10)

X = X.astype("float32") / 255.0

# Encode genres properly now
label_encoder = LabelEncoder()
y = [str(label).strip() for label in y]  # clean up
y_encoded = label_encoder.fit_transform(y)

#  Check for label explosion
num_classes = len(label_encoder.classes_)
if num_classes > 500:
    print("Top 10 classes:", label_encoder.classes_[:10])
    raise ValueError("Too many unique classes (>500). Check your labels!")

y_categorical = to_categorical(y_encoded)

base_model = EfficientNetB0(include_top=False, input_shape=(64, 64, 3), weights='imagenet')
base_model.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

print(" Starting training...")
history = model.fit(X, y_categorical, epochs=5, batch_size=32, validation_split=0.2)

model.save(os.path.join(PROJECT_DIR, "album_movie_classifier.h5"))
np.save(os.path.join(PROJECT_DIR, "class_labels.npy"), label_encoder.classes_)
print(" Model and labels saved!")
