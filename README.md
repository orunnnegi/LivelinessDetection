# Anti-Spoofing / Liveliness Detection for Face Recognition Systems
---
## Overview
This project enhances the security of face recognition systems by developing an Anti-Spoofing/Liveliness Detector. The solution distinguishes between real and simulated facial appearances, preventing spoofing attacks such as printed images or virtual representations. It is particularly useful for automated attendance systems in offices and schools.
---
## Table of Contents
- [Introduction](#introduction)
- [Problem Statement](#problem-statement)
- [Project Goals](#project-goals)
- [Significance](#significance)
- [Key Takeaways](#key-takeaways)
- [Literature Survey](#literature-survey)
- [Methodology](#methodology)
  - [Data Collection](#data-collection)
  - [Data Splitting](#data-splitting)
  - [Data Configuration](#data-configuration)
  - [Training Offline](#training-offline)
  - [Real-Time Inference](#real-time-inference)
  - [Performance Metrics and Result Display](#performance-metrics-and-result-display)
- [Results and Discussion](#results-and-discussion)
- [Conclusion and Future Work](#conclusion-and-future-work)
- [How to Clone and Use the Project](#how-to-clone-and-use-the-project)
- [References](#references)
---
## Introduction
In computer vision and face recognition, technological advancements have dramatically improved convenience and efficiency. However, these systems face challenges related to security, particularly spoofing attempts where fake images or visual representations are used to deceive the system. This project addresses that vulnerability by detecting and differentiating between genuine and spoofed facial inputs.
---
## Problem Statement
The primary objective is to secure face recognition systems by preventing spoofing attempts. The system must differentiate between authentic facial inputs and deceptive ones, whether they come as printed images, virtual representations, or manipulated visuals.
---
## Project Goals
- **Enhanced Security:** Integrate robust anti-spoofing mechanisms to secure face recognition systems.
- **Streamlined Attendance:** Automate attendance processes in environments like offices and schools by enabling simultaneous detection of multiple faces.
- **Efficient Dataset Creation:** Develop a custom dataset rapidly, consisting of thousands of images (both genuine and spoofed), for effective model training.
---
## Significance
The project is significant because it:
- Provides a practical approach to dataset collection and training without over-reliance on pre-existing online datasets.
- Equips developers with the skills needed to adapt models to the specific characteristics of their environment.
- Advances research in anti-spoofing techniques for real-time face recognition systems.
---
## Key Takeaways
- **Rapid Dataset Creation:** Generate a diverse and representative dataset quickly.
- **Model Adaptability:** Fine-tune models to respond effectively to evolving spoofing techniques.
- **Real-Time Application:** Demonstrate the feasibility of real-time anti-spoofing detection through integration with live video feeds.
---
## Literature Survey
Research in face recognition has achieved high accuracy with deep neural networks, but many systems remain vulnerable to spoofing. Key insights include:
- Deep learning models excel at feature extraction yet often overlook spoofing challenges.
- Robust anti-spoofing methods are necessary to counteract the security challenges in face recognition.
- Multi-faceted approaches combining texture analysis and dynamic facial cues are essential.
- Custom-tailored datasets prove more effective than relying solely on publicly available data.
- Real-time systems must balance speed and accuracy to be practically deployable.
---
## Methodology

### Data Collection
- **Capture:** Video frames are captured from a webcam.
- **Detection:** Uses the `cvzone.FaceDetectionModule` for face detection.
- **Classification:** Faces are classified as real or spoofed based on a blurriness measure using Laplacian variance.
- **Labeling:** Normalizes face bounding box coordinates and saves images with label information.

### Data Splitting
- **Organization:** The collected data is split into training, validation, and test sets based on predetermined ratios.
- **Structure:** Separate directories are created for each set to organize images and labels.

### Data Configuration
- **File Generation:** A `data.yaml` file is created with paths to training, validation, and test sets, the number of classes, and class names.

### Training Offline
- **Model:** Utilizes the YOLO (You Only Look Once) object detection model from the Ultralytics library.
- **Pre-Trained:** Uses a pre-trained YOLO model (`yolov8n.pt`) for training over several epochs with the custom dataset.

### Real-Time Inference
- **Live Feed:** Captures live video feed from a webcam.
- **Detection:** Uses the trained YOLO model to detect faces and classify them as real or spoofed.
- **Visualization:** Displays bounding boxes around faces (green for real, red for fake) along with confidence scores and labels.

### Performance Metrics and Result Display
- **Evaluation:** The system is evaluated using metrics such as accuracy, precision, recall, and F1 score.
- **Frame Rate:** Measures real-time frame rate (fps) for efficient processing.
- **Feedback:** Processed video frames are displayed with real-time feedback.

## Results and Discussion
The system successfully:
- **Detects in Real-Time:** Accurately identifies and differentiates between real and spoofed faces in a live video feed.
- **Provides Visual Feedback:** Uses color-coded bounding boxes and confidence scores to indicate detection results.
- **Performs Robustly:** Standard metrics indicate reliable performance, though factors like lighting and dataset quality can affect accuracy.

## Conclusion and Future Work
The Anti-Spoofing/Liveliness Detector significantly enhances face recognition security by effectively differentiating between real and fake faces. Key learnings include improvements in OpenCV, data collection, and model training. Future work may involve:
- Expanding the dataset with more diverse images.
- Exploring additional anti-spoofing techniques and further model fine-tuning.
- Deploying and testing the system in varied real-world environments.
---
## How to Clone and Use the Project

### Cloning the Repository
1. **Open Terminal or Command Prompt:**
   Navigate to the directory where you want to clone the project.
   
2. **Clone the Repository:**
   ```bash
   git clone https://github.com/orunnnegi/LivelinessDetection.git
3. **Navigate into the Project Directory:**

```bash
cd LivelinessDetection
```

4. **Setting Up the Environment:**

Install Dependencies: Ensure you have Python installed. Then install the required packages using:

```bash

pip install -r requirements.txt
```

This will install OpenCV, cvzone, and any other packages specified.

---

## Prepare the Dataset:

1. **Data Collection: Run the data collection script to capture and label images. Ensure your webcam is connected.**

2. **Data Splitting: Execute the script for splitting data into training, validation, and test sets.**

3. **Running the Project.**

4. **Training the Model:**

5. **Run the training script to start model training with your custom dataset:**


```bash
python train.py
```

6. **Real-Time Inference:**

After training, execute the real-time inference script to see the detection in action:


```bash
python app.py
```
The system will activate your webcam and display real-time feedback on the classification of faces as real or spoofed.


---

### Additional Configuration:

**Update the data.yaml file if needed to adjust paths or class details.**

**Tweak model parameters in the training script based on your requirements.**





