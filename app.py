import cv2
import numpy as np
from deepface import DeepFace
import threading
import tkinter as tk

# Load the Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize mode state
mode = "blur_all"  # Default mode is blur all
face_cache = {}  # Cache for storing age predictions and frame count
frame_count = 0  # Global frame count for skipping

def process_faces(image):
    global face_cache, frame_count

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5, minSize=(30, 30))

    # Process each detected face
    for (x, y, w, h) in faces:
        # Ensure the coordinates are within the image bounds
        x_start = max(0, x)
        y_start = max(0, y)
        x_end = min(image.shape[1], x + w)
        y_end = min(image.shape[0], y + h)

        # Extract the face region for age prediction
        face_region = image[y_start:y_end, x_start:x_end]
        key = (x, y, w, h)  # Cache key based on face location

        # Initialize age variable
        age = None

        # Process based on mode
        if mode in ["blur_all", "blur_under_18"]:  # Blur faces if in blur mode
            if mode == "blur_under_18":
                if key in face_cache and (frame_count - face_cache[key]['frame']) < 10:  # Cache age for 10 frames
                    age = face_cache[key]['age']
                else:
                    try:
                        # Use DeepFace to predict age
                        analysis = DeepFace.analyze(face_region, actions=['age'], enforce_detection=False)
                        age = analysis[0]['age']

                        # Update cache with the new age prediction
                        face_cache[key] = {'age': age, 'frame': frame_count}

                    except Exception as e:
                        print(f"Error analyzing face: {e}")
                        age = 100  # Default to a high age if analysis fails

                # Blur if under 18
                if age is not None and age < 18:
                    blurred_face = cv2.GaussianBlur(face_region, (99, 99), 30)
                    image[y_start:y_end, x_start:x_end] = blurred_face

            elif mode == "blur_all":  # Blur all faces
                blurred_face = cv2.GaussianBlur(face_region, (99, 99), 30)
                image[y_start:y_end, x_start:x_end] = blurred_face

        elif mode in ["pixelate_all", "pixelate_under_18"]:  # Pixelate faces if in pixelate mode
            if mode == "pixelate_under_18":
                if key in face_cache and (frame_count - face_cache[key]['frame']) < 10:  # Cache age for 10 frames
                    age = face_cache[key]['age']
                else:
                    try:
                        # Use DeepFace to predict age
                        analysis = DeepFace.analyze(face_region, actions=['age'], enforce_detection=False)
                        age = analysis[0]['age']

                        # Update cache with the new age prediction
                        face_cache[key] = {'age': age, 'frame': frame_count}

                    except Exception as e:
                        print(f"Error analyzing face: {e}")
                        age = 100  # Default to a high age if analysis fails

                # If age is under 18, pixelate the face
                if age is not None and age < 18:
                    small_face = cv2.resize(face_region, (10, 10))  # Resize to pixelate
                    pixelated_face = cv2.resize(small_face, (w, h), interpolation=cv2.INTER_NEAREST)
                    image[y_start:y_end, x_start:x_end] = pixelated_face

            elif mode == "pixelate_all":  # Pixelate all faces
                small_face = cv2.resize(face_region, (10, 10))  # Resize to pixelate
                pixelated_face = cv2.resize(small_face, (w, h), interpolation=cv2.INTER_NEAREST)
                image[y_start:y_end, x_start:x_end] = pixelated_face

        # Draw bounding box and age label
        if age is not None:
            age_text = f"{age} years"
            # Draw bounding box
            cv2.rectangle(image, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
            # Draw age label above the bounding box
            cv2.putText(image, age_text, (x_start, y_start - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image

def video_loop(video_capture):
    global frame_count

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        # Increment frame count
        frame_count += 1

        # Process the frame
        frame_with_effect = process_faces(frame)

        # Display the resulting frame
        cv2.imshow('Face Blur/Pixellate', frame_with_effect)

        # Wait for key input
        key = cv2.waitKey(1) & 0xFF
        
        # Press 'q' to exit
        if key == ord('q'):
            break

def set_mode(new_mode):
    global mode
    mode = new_mode
    print(f"Mode set to: {mode}")

def create_ui():
    # Create the main window
    window = tk.Tk()
    window.title("Face Processing Modes")

    # Create buttons for each mode
    tk.Button(window, text="Blur All", command=lambda: set_mode("blur_all")).pack(pady=10)
    tk.Button(window, text="Blur Under 18", command=lambda: set_mode("blur_under_18")).pack(pady=10)
    tk.Button(window, text="Pixelate All", command=lambda: set_mode("pixelate_all")).pack(pady=10)
    tk.Button(window, text="Pixelate Under 18", command=lambda: set_mode("pixelate_under_18")).pack(pady=10)
    tk.Button(window, text="No Effect", command=lambda: set_mode("no_effect")).pack(pady=10)
    tk.Button(window, text="Exit", command=window.quit).pack(pady=10)

    # Run the Tkinter event loop
    window.mainloop()

def main():
    global mode

    # Load the webcam
    video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use DirectShow for Windows

    if not video_capture.isOpened():
        print("Error: Cannot open camera")
        return

    # Start video processing in a separate thread
    threading.Thread(target=video_loop, args=(video_capture,), daemon=True).start()

    # Start the Tkinter UI
    create_ui()

    # Release the webcam and close windows
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
