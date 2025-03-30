<<<<<<< HEAD
import streamlit as st
import cv2
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from ultralytics import YOLO
import cvzone
from PIL import Image

# Initialize the model and other variables


model = YOLO("C:/Liveliness Detection/models/myModel.pt")
classNames = ["fake", "real"]
confidence_threshold = 0.8
cap = None

# Streamlit state management

if 'running' not in st.session_state: # to check if the camera is not running by default
    st.session_state.running = False
if 'data' not in st.session_state: # to check if already data is present in the real and fake keys in the dictionary
    st.session_state.data = {'real': [], 'fake': []}
if 'fake_snapshots' not in st.session_state:
    st.session_state.fake_snapshots = []

# Functions
def start_camera():
    global cap
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)
    st.session_state.running = True

def stop_camera():
    global cap
    if cap is not None:
        cap.release()
        cap = None
    st.session_state.running = False
    show_report()
    save_fake_collage()

def update_frame():
    global cap
    ret, img = cap.read()
    if not ret:
        return None, 0, 0, 0

    results = model(img, stream=True, verbose=False)
    real_count = 0
    fake_count = 0

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            conf = box.conf[0]
            cls = int(box.cls[0])
            if conf > confidence_threshold:
                if classNames[cls] == 'real':
                    color = (0, 255, 0)
                    real_count += 1
                else:
                    color = (0, 0, 255)
                    fake_count += 1
                    st.session_state.fake_snapshots.append(img.copy())

                cvzone.cornerRect(img, (x1, y1, w, h), colorC=color, colorR=color)
                cvzone.putTextRect(img, f'{classNames[cls].upper()} {int(conf * 100)}%',
                                   (max(0, x1), max(35, y1)), scale=2, thickness=4, colorR=color,
                                   colorB=color)

    st.session_state.data['real'].append(real_count)
    st.session_state.data['fake'].append(fake_count)

    fps = cap.get(cv2.CAP_PROP_FPS)
    return img, real_count, fake_count, fps

def show_report():
    total_real = sum(st.session_state.data['real'])
    total_fake = sum(st.session_state.data['fake'])
    st.write(f"Total Real Detections: {total_real}")
    st.write(f"Total Fake Detections: {total_fake}")

    plt.figure(figsize=(8, 6))
    plt.bar(['Real', 'Fake'], [total_real, total_fake], color=['green', 'red'])
    plt.xlabel('Detection Type')
    plt.ylabel('Count')
    plt.title('Detection Summary')
    st.pyplot(plt)

def save_fake_collage():
    if not st.session_state.fake_snapshots:
        st.write("No fake snapshots available to create collage.")
        return

    rows = int(np.sqrt(len(st.session_state.fake_snapshots)))
    cols = int(np.ceil(len(st.session_state.fake_snapshots) / rows))
    thumbnail_size = (200, 200)
    collage_width = cols * thumbnail_size[0]
    collage_height = rows * thumbnail_size[1]

    collage = np.zeros((collage_height, collage_width, 3), dtype=np.uint8)

    for idx, snapshot in enumerate(st.session_state.fake_snapshots):
        row = idx // cols
        col = idx % cols
        thumbnail = cv2.resize(snapshot, thumbnail_size)
        y1, y2 = row * thumbnail_size[1], (row + 1) * thumbnail_size[1]
        x1, x2 = col * thumbnail_size[0], (col + 1) * thumbnail_size[0]
        collage[y1:y2, x1:x2] = thumbnail

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    collage_filename = f'snapshots/collection_fake_images_{timestamp}.png'
    cv2.imwrite(collage_filename, collage)
    st.write(f"Collage of fake snapshots saved as {collage_filename}.")
    st.image(collage, channels="BGR")

# Streamlit interface
st.title("Anti-Spoofing Detection System")
st.write("Use the buttons below to start and stop the camera.")

start_button = st.button('Start Camera')
stop_button = st.button('Stop Camera')

if start_button:
    start_camera()

if stop_button:
    stop_camera()

if st.session_state.running:
    frame_placeholder = st.empty()
    while st.session_state.running:
        img, real_count, fake_count, fps = update_frame()
        if img is not None:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(img_rgb, channels="RGB")
        time.sleep(1 / fps)
=======
import streamlit as st
import cv2
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from ultralytics import YOLO
import cvzone
from PIL import Image

# Initialize the model and other variables


model = YOLO("C:/Liveliness Detection/models/myModel.pt")
classNames = ["fake", "real"]
confidence_threshold = 0.8
cap = None

# Streamlit state management

if 'running' not in st.session_state: # to check if the camera is not running by default
    st.session_state.running = False
if 'data' not in st.session_state: # to check if already data is present in the real and fake keys in the dictionary
    st.session_state.data = {'real': [], 'fake': []}
if 'fake_snapshots' not in st.session_state:
    st.session_state.fake_snapshots = []

# Functions
def start_camera():
    global cap
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)
    st.session_state.running = True

def stop_camera():
    global cap
    if cap is not None:
        cap.release()
        cap = None
    st.session_state.running = False
    show_report()
    save_fake_collage()

def update_frame():
    global cap
    ret, img = cap.read()
    if not ret:
        return None, 0, 0, 0

    results = model(img, stream=True, verbose=False)
    real_count = 0
    fake_count = 0

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            conf = box.conf[0]
            cls = int(box.cls[0])
            if conf > confidence_threshold:
                if classNames[cls] == 'real':
                    color = (0, 255, 0)
                    real_count += 1
                else:
                    color = (0, 0, 255)
                    fake_count += 1
                    st.session_state.fake_snapshots.append(img.copy())

                cvzone.cornerRect(img, (x1, y1, w, h), colorC=color, colorR=color)
                cvzone.putTextRect(img, f'{classNames[cls].upper()} {int(conf * 100)}%',
                                   (max(0, x1), max(35, y1)), scale=2, thickness=4, colorR=color,
                                   colorB=color)

    st.session_state.data['real'].append(real_count)
    st.session_state.data['fake'].append(fake_count)

    fps = cap.get(cv2.CAP_PROP_FPS)
    return img, real_count, fake_count, fps

def show_report():
    total_real = sum(st.session_state.data['real'])
    total_fake = sum(st.session_state.data['fake'])
    st.write(f"Total Real Detections: {total_real}")
    st.write(f"Total Fake Detections: {total_fake}")

    plt.figure(figsize=(8, 6))
    plt.bar(['Real', 'Fake'], [total_real, total_fake], color=['green', 'red'])
    plt.xlabel('Detection Type')
    plt.ylabel('Count')
    plt.title('Detection Summary')
    st.pyplot(plt)

def save_fake_collage():
    if not st.session_state.fake_snapshots:
        st.write("No fake snapshots available to create collage.")
        return

    rows = int(np.sqrt(len(st.session_state.fake_snapshots)))
    cols = int(np.ceil(len(st.session_state.fake_snapshots) / rows))
    thumbnail_size = (200, 200)
    collage_width = cols * thumbnail_size[0]
    collage_height = rows * thumbnail_size[1]

    collage = np.zeros((collage_height, collage_width, 3), dtype=np.uint8)

    for idx, snapshot in enumerate(st.session_state.fake_snapshots):
        row = idx // cols
        col = idx % cols
        thumbnail = cv2.resize(snapshot, thumbnail_size)
        y1, y2 = row * thumbnail_size[1], (row + 1) * thumbnail_size[1]
        x1, x2 = col * thumbnail_size[0], (col + 1) * thumbnail_size[0]
        collage[y1:y2, x1:x2] = thumbnail

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    collage_filename = f'snapshots/collection_fake_images_{timestamp}.png'
    cv2.imwrite(collage_filename, collage)
    st.write(f"Collage of fake snapshots saved as {collage_filename}.")
    st.image(collage, channels="BGR")

# Streamlit interface
st.title("Anti-Spoofing Detection System")
st.write("Use the buttons below to start and stop the camera.")

start_button = st.button('Start Camera')
stop_button = st.button('Stop Camera')

if start_button:
    start_camera()

if stop_button:
    stop_camera()

if st.session_state.running:
    frame_placeholder = st.empty()
    while st.session_state.running:
        img, real_count, fake_count, fps = update_frame()
        if img is not None:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(img_rgb, channels="RGB")
        time.sleep(1 / fps)
>>>>>>> 0058476 (Initial commit of Liveliness Detection project)
