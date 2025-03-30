<<<<<<< HEAD
import streamlit as st
import cv2
import math
import time
import cvzone
from ultralytics import YOLO

confidence = 0.8   # Sets a threshold for the confidence score.
# Only detections with confidence above this value will be considered.

model = YOLO("models/myModel.pt")
classNames = ["fake", "real"]

# Initialize camera capture
cap = None

# Function to start the camera
def start_camera():
    global cap
    cap = cv2.VideoCapture(0)  # For Webcam
    cap.set(3, 640)
    cap.set(4, 480)   # Sets the width and height of the video frame.

# Function to stop the camera
def stop_camera():
    global cap
    if cap is not None:
        cap.release()
        cap = None

def main():
    st.title("Live Object Detection with Streamlit")

    # Button to start the camera
    if st.button("Start Camera"):
        start_camera()

    # Button to stop the camera
    if st.button("Stop Camera"):
        stop_camera()

    # Display camera feed and perform object detection
    if cap is not None:
        prev_frame_time = 0
        while cap.isOpened():
            new_frame_time = time.time()    # Records the current time for FPS calculation.
            success, img = cap.read()       # Reads a frame from the webcam.
            if not success:
                st.warning("Failed to capture frame from camera.")
                break

            results = model(img, stream=True, verbose=False)  # Runs the yolo model on the captured frame, get detection result
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    w, h = x2 - x1, y2 - y1
                    conf = math.ceil((box.conf[0] * 100)) / 100
                    cls = int(box.cls[0])
                    if conf > confidence:
                        if classNames[cls] == 'real':
                            color = (0, 255, 0)
                        else:
                            color = (0, 0, 255)

                        cvzone.cornerRect(img, (x1, y1, w, h), colorC=color, colorR=color)
                        cvzone.putTextRect(img, f'{classNames[cls].upper()} {int(conf*100)}%',
                                           (max(0, x1), max(35, y1)), scale=2, thickness=4, colorR=color,
                                           colorB=color)

            fps = 1 / (new_frame_time - prev_frame_time)
            prev_frame_time = new_frame_time
            st.image(img, channels="BGR")  # Display the processed frame

            st.write(f"FPS: {fps:.2f}")

            # Check if 'Stop Camera' button was pressed
            if not st.session_state.stop_camera_button:
                break

if __name__ == "__main__":
=======
import streamlit as st
import cv2
import math
import time
import cvzone
from ultralytics import YOLO

confidence = 0.8   # Sets a threshold for the confidence score.
# Only detections with confidence above this value will be considered.

model = YOLO("models/myModel.pt")
classNames = ["fake", "real"]

# Initialize camera capture
cap = None

# Function to start the camera
def start_camera():
    global cap
    cap = cv2.VideoCapture(0)  # For Webcam
    cap.set(3, 640)
    cap.set(4, 480)   # Sets the width and height of the video frame.

# Function to stop the camera
def stop_camera():
    global cap
    if cap is not None:
        cap.release()
        cap = None

def main():
    st.title("Live Object Detection with Streamlit")

    # Button to start the camera
    if st.button("Start Camera"):
        start_camera()

    # Button to stop the camera
    if st.button("Stop Camera"):
        stop_camera()

    # Display camera feed and perform object detection
    if cap is not None:
        prev_frame_time = 0
        while cap.isOpened():
            new_frame_time = time.time()    # Records the current time for FPS calculation.
            success, img = cap.read()       # Reads a frame from the webcam.
            if not success:
                st.warning("Failed to capture frame from camera.")
                break

            results = model(img, stream=True, verbose=False)  # Runs the yolo model on the captured frame, get detection result
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    w, h = x2 - x1, y2 - y1
                    conf = math.ceil((box.conf[0] * 100)) / 100
                    cls = int(box.cls[0])
                    if conf > confidence:
                        if classNames[cls] == 'real':
                            color = (0, 255, 0)
                        else:
                            color = (0, 0, 255)

                        cvzone.cornerRect(img, (x1, y1, w, h), colorC=color, colorR=color)
                        cvzone.putTextRect(img, f'{classNames[cls].upper()} {int(conf*100)}%',
                                           (max(0, x1), max(35, y1)), scale=2, thickness=4, colorR=color,
                                           colorB=color)

            fps = 1 / (new_frame_time - prev_frame_time)
            prev_frame_time = new_frame_time
            st.image(img, channels="BGR")  # Display the processed frame

            st.write(f"FPS: {fps:.2f}")

            # Check if 'Stop Camera' button was pressed
            if not st.session_state.stop_camera_button:
                break

if __name__ == "__main__":
>>>>>>> 0058476 (Initial commit of Liveliness Detection project)
    main()