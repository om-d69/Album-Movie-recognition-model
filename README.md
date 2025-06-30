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

>  Current classifier accuracy is limited by genre label complexity and class overlaps. Future improvements will involve label simplification and deeper architectures (EfficientNet with Transfer Learning).


**File Structure**:
