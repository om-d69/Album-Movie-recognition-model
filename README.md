#  Movie Poster Recognition & IMDb Link Opener (Real-Time CV + ML Project)

This is a real-time **Movie Poster Recognition System** that uses **OpenCV + TensorFlow** to identify posters shown on a webcam (e.g., on your phone screen), predict the associated movie genre or title using a trained CNN model, and automatically open the corresponding **IMDb link** in your browser.

>  **Note:** This version focuses only on **movie posters**. Music album support and model accuracy improvements are planned in a future release.

---

##  Project Highlights

-  Uses a **CNN model trained on Kaggle's "Movie Posters by Raman" dataset**
-  Real-time webcam integration using **OpenCV**
-  Model trained with **TensorFlow/Keras** on resized 64×64 poster images
-  Predicts **top genre/class** based on poster input and opens its IMDb link
-  Simulates Face ID–style green box detection for poster targeting
-  All model and dataset assets stored and processed in local directory
-  Functional despite imperfect prediction accuracy
-  GUI and Spotify album support planned for future iterations

---

##  Tech Stack Used         
 Language: Python 3.11           
 ML Framework: TensorFlow 2 / Keras  
 CV Library: OpenCV                
 Data Handling: Pandas, NumPy         
 Image Preprocessing: `cv2`, `img_to_array` 
 Dataset Source: [Kaggle - Movie Posters by Raman](https://www.kaggle.com/datasets/raman7777/movie-posters) 
 IDE: VS Code (Local)       
 OS: Windows 11            

---

##  Model Architecture

- **Base:** Custom CNN with `Conv2D`, `MaxPooling`, `Flatten`, and `Dense` layers
- **Input Size:** 64×64 RGB Poster Images
- **Activation:** ReLU + Softmax
- **Optimizer:** Adam
- **Loss Function:** Categorical Crossentropy
- **Final Output:** Softmax scores over top N classes
- **ResNet50** and **EfficientNetB0** were evaluated during model development.
   - Both were used as feature extractors using ImageNet weights and custom classification heads.
   - Ultimately, due to simplicity, speed, and integration reasons, a custom CNN was chosen for final deployment.

 Trained model: `album_movie_classifier.h5`

>  Current classifier accuracy is limited by genre label complexity and class overlaps. Future improvements will involve label simplification and deeper architectures (EfficientNet with Transfer Learning).

Sure! Here's your neatly formatted and structured project procedure, exactly as you provided but cleaned up for clarity and presentation:

---

#  Movie Poster Genre Classification using CNN + OpenCV

**Author:** Om Deshpande
**Program:** B.Tech Cyber Physical Systems
**Institute:** Manipal Institute of Technology

---

##  PROCEDURE FOR THIS PROJECT FROM THE GROUND UP

### **STEP 1: Dataset Collection and Curation**

**Objective:** Collect and organize the raw dataset of movie posters and prepare basic metadata.
**Details:**

* **Source:** Kaggle Dataset – *"Movie Posters by Raman"*
* **File types:**

  * `images/` folder containing `.jpg` or `.png` poster files
  * `train.csv` or `MoviePosters_Raman_fixed.csv` containing:

    * `filename`: Name of the image file
    * `genre`: A comma-separated string of genres like `"Action, Drama, Thriller"`
      **Outcome:** Dataset folder and metadata CSVs are ready for preprocessing.

---

### **STEP 2: Fixing the Dataset Labels**

**Objective:** Clean, normalize, and reduce the number of classes.
**Details:**

* Fix inconsistencies in the `genre` column (extra spaces, casing, missing values)
* Cleaned CSV file (e.g., `train_fixed.csv` or `movie_with_links.csv`) should include:

  * `genre` combinations as label
  * `imdb_link` column with direct IMDb links
    **Outcome:** A clean, minimal CSV ready for preprocessing and label encoding.

---

### **STEP 3: Image Preprocessing & Saving Batches**

**Objective:** Preprocess images and save them into `.npy` batches for efficient training.
**Details:**

* Read each image from the `images/` folder using filenames in the CSV
* Resize images to **64×64**
* Convert images to NumPy arrays and save them in batches like:

  * `images_batch_movie_0.npy`
  * `labels_batch_movie_0.npy`
* Each label corresponds to the genre string
  **Outcome:** Efficient `.npy` files with image and label arrays ready for model training.

---

### **STEP 4: CNN Model Training**

**Objective:** Train a CNN model on the preprocessed image and label batches.
**Details:**

* Load `.npy` image and label batches from Step 3
* Encode genre strings into integers using `LabelEncoder`
* One-hot encode labels using `to_categorical()`
* Define and compile a CNN model:
  `Conv2D → MaxPool → Flatten → Dense`
* Train and save model as:

  * `album_movie_classifier.h5`
    **Outcome:** A trained `.h5` model to classify movie poster genres.

---

### **STEP 5: Real-Time Poster Prediction via Webcam**

**Objective:** Use webcam to detect a poster and display the top predicted genre.
**Details:**

* Open webcam using `cv2.VideoCapture(0)`
* Define a **green Face ID–style box** at the center of the frame
* Capture the image inside the box and resize to **64×64**
* Predict using the trained CNN model
* Display predicted genre and confidence on screen (if above threshold)
  **Outcome:** Real-time webcam classification of posters when held in front of the screen.

---

### **STEP 6: IMDb Link Opening + Face Box Integration**

**Objective:** Extend Phase 5 by opening a browser tab to the corresponding IMDb link.
**Details:**

* Use prediction to match the genre string in `movie_with_links.csv`
* Retrieve the `imdb_link` for the predicted genre
* Open IMDb link using `webbrowser.open()` (trigger only once per detection)
* Add a **10–15 second delay** before prediction to allow time to place the poster
* Avoid repeated link openings unless the prediction changes
  **Outcome:** Real-time poster detection, genre classification, and IMDb redirection.

---

##  FILE STRUCTURE (Ordered)

* **`phase1_cam_feed.py`** – Initializes and tests webcam access using OpenCV
* **`phase2.py`** – Loads ResNet50 and performs real-time classification using ImageNet
* **`phase3.py`** – Preprocesses dataset, cleans labels, resizes images, saves `.npy` batches
* **`phase4.py`** – Loads batches for verification and debugging (shapes, labels)
* **`train_model.py`** – Trains EfficientNetB0 on `.npy` files, saves `.h5` model and label mappings
* *(Optional)* **`image_classifier.py`** – Tests image classification using the trained model
* **`webcam_predictor_phase6.py`** – Real-time genre prediction with motion detection and green box
* **`gui_phase7.py`** – Tkinter GUI showing prediction, clickable IMDb link, and delay setup

---

##  LIMITATIONS AND FUTURE WORKINGS

### **Known Limitations:**

* **Low Accuracy:** Model often outputs same genre (e.g., "Drama") due to class imbalance
* **IMDb Link Mismatch:** Only first match of genre is used (can be inaccurate)
* **Poster-Only Support:** Faces, scenery, and blurred images are not detected
* **Low Resolution (64×64):** Limits detail; higher resolution could improve performance

### **Planned Improvements:**

* Enhance classification via **multi-label encoding** or **title recognition**
* Add support for **Spotify album recognition** with separate model
* Integrate **Tkinter GUI** with clickable links and thumbnails
* Use **YOLOv8/MediaPipe** for precise poster detection
* Optimize model for **low-light webcam** and **mobile compatibility**

