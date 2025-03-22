import cv2
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image


model = ResNet50(weights='imagenet')

cap = cv2.VideoCapture(0)


if not cap.isOpened():
    print("Error: Camera not accessible.")
    exit()


def predict_frame(frame):
    try:
       
        img = cv2.resize(frame, (224, 224))

       
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

       
        img_array = image.img_to_array(img_rgb)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        
        preds = model.predict(img_array)

       
        predictions = decode_predictions(preds, top=2)[0]

       
        filtered_preds = [(label, confidence) for (_, label, confidence) in predictions if confidence > 0.5]

        return filtered_preds
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None


print("Press 'q' to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to read frame from camera.")
        break

   
    predictions = predict_frame(frame)

    
    if predictions:
        print("Predictions:")
        for label, confidence in predictions:
            print(f"  {label}: {confidence * 100:.2f}%")

   
    if predictions:
        y_offset = 30  
        for label, confidence in predictions:
            text = f"{label}: {confidence * 100:.2f}%"
            cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y_offset += 30  

    
    cv2.imshow("Camera Feed with Predictions", frame)

   
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
