import cv2

def main():
    cap = cv2.VideoCapture(0)  

    if not cap.isOpened():
        print("Error: Camera not accessible.")
        return
    
    print("press 's' to save an image, or 'q' to quit.")

    while True:
        ret, frame = cap.read()  
        if not ret:
            print("Failed to grab frame.")
            break

        cv2.imshow("Camera Feed", frame)  


        key=cv2.waitKey(1) & 0xFF 
        if key == ord('q'): 
            break
        elif key == ord('s'):
            filename = "captured_image.jpg"
            cv2.imwrite(filename, frame)
            print(f"image saved as {filename}")

    cap.release()
    cv2.destroyAllWindows()  

if __name__ == "__main__":
    main()
