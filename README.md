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

**PROCEDURE FOR THIS PROJECT FROM THE GROUND UP**

**STEP1 1**: **Dataset Collection and Curation**
Objective: Collect and organize the raw dataset of movie posters and prepare basic metadata.
Details:
Source: Kaggle Dataset – "Movie Posters by Raman"
File types:
images/ folder containing .jpg or .png poster files.
train.csv or MoviePosters_Raman_fixed.csv containing two key columns:
filename – Name of the image file.
genre – A comma-separated string of genres like "Action, Drama, Thriller".
Outcome: You now have a dataset folder and metadata CSVs that will feed into preprocessing.

**STEP 2**: **Fixing the Dataset Labels**
Objective: Clean, normalize, or reduce the number of classes.
Details:
Fix inconsistencies in the genre column (extra spaces, casing, missing values).
Clean a CSV like train_fixed.csv or movie_with_links.csv with:
Genre combinations as label.
include direct IMDb links (imdb_link column).
Outcome: A clean, minimal CSV ready for preprocessing and label encoding.

**STEP 3**: **Image Preprocessing & Saving Batches**
Objective: Preprocess images and save them into .npy batches for efficient training.
Details:
Reads each image from images/ folder based on filenames in the CSV.
Resizes all images to 64×64 (as required by the CNN input).
Converts the images into NumPy arrays and stores them in batches like:
images_batch_movie_0.npy
labels_batch_movie_0.npy
Each label corresponds to the genre string.
Outcome: Efficient .npy files containing both images and label arrays for model training.

**STEP 4**: **CNN Model Training**
Objective: Train a CNN model on the preprocessed image and label batches.
Details:
Loads image batches and label batches from Phase 3.
Encodes genre strings into integer labels using LabelEncoder.
Uses to_categorical() to one-hot encode the labels.
Defines and compiles a CNN model (Conv2D → MaxPool → Flatten → Dense).
Trains the model and saves it to:
album_movie_classifier.h5
Outcome: A trained Keras .h5 model that can classify movie poster genres.

**STEP 5**: **Real-Time Poster Prediction via Webcam**
Objective: Use webcam to detect a poster and display the top predicted genre.
Details:
Opens webcam using cv2.VideoCapture(0)
Defines a green Face ID–style box at the center of the frame.
Captures the image inside that box and resizes it to 64×64.
Predicts using the CNN model from Phase 4.
Displays the predicted genre and confidence on screen if above threshold.
Outcome: Real-time webcam classification of posters when held in front of the screen.

**STEP 6**: **IMDb Link Opening + Face Box Integration**
Objective: Extend Phase 5 by opening a web browser to show the IMDb link for the predicted genre/movie.
Details:
Uses prediction from the CNN model to find the correct genre string.
Looks up this genre in the CSV (movie_with_links.csv) to find the associated IMDb link.
Opens the IMDb link using webbrowser.open() (once per detection).
Introduces a 10–15 second delay before triggering prediction to let you position your phone/poster inside the green box.
Adds logic to prevent repeated link openings unless the prediction changes.
Outcome: Real-time genre classification AND browser redirection to IMDb — movie poster.

**FILE STRUCTURE DEFINED(ordered):**
phase1_cam_feed.py: Initializes and tests your system’s webcam using OpenCV to confirm camera access.
phase2.py: Loads the ResNet50 model and performs real-time image classification using webcam frames with basic predictions from the ImageNet dataset.
phase3.py: Preprocesses the MoviePosters_Raman dataset: reads the CSV, cleans up labels, resizes images, and saves them in batch .npy files for efficient training.
phase4.py: Loads and previews image batches saved from Phase 3 to confirm correct formatting, shapes, and label associations. Mostly for verification and debugging.
train_model.py: Trains an EfficientNetB0 model on the preprocessed image batches and saves the final trained model (.h5) and encoded class labels (.npy).
(optional): image_classifier.py: Tests image classification
webcam_predictor_phase6.py: Opens your webcam and uses motion detection to trigger genre predictions using the trained model. Draws a green box on the region of interest and overlays the predicted class with confidence.
gui_phase7.py: Displaying predictions and clickable links (IMDb) and includes delayed link opening + face-like box behavior.
