import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam


PROJECT_DIR = r"C:\Users\omdes\Documents\Om\opencvfun\opencv_album_recog_proj"

X1 = np.load(os.path.join(PROJECT_DIR, "images_batch_album_0.npy"), allow_pickle=True)
Y1 = np.load(os.path.join(PROJECT_DIR, "labels_batch_album_0.npy"), allow_pickle=True)

X2 = np.load(os.path.join(PROJECT_DIR, "images_batch_album_1.npy"),allow_pickle=True)
Y2 = np.load(os.path.join(PROJECT_DIR, "labels_batch_album_1.npy"),allow_pickle=True)

X = np.concatenate((X1, X2), axis = 0)
y = np.concatenate((Y1, Y2), axis = 0) 

X = X.astype("float32")/255.0

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)          
y_categorical = to_categorical(y_encoded)

base_model  = EfficientNetB0(include_top = False, input_shape = (64, 64, 3), weights = 'imagenet')

base_model.trainable = False

model = Sequential([base_model, GlobalAveragePooling2D(),Dense(128, activation = 'relu'), Dense(len(label_encoder.classes_), activation='softmax')])

model.compile(optimizer = Adam(learning_rate = 0.001), loss= 'categorical_crossentropy', metrics = ['accuracy'])


print("correct data shape", X.shape, "| Labels:", y[:5])
print("Correct Classes:", label_encoder.classes_)

model.summary()