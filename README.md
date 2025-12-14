# Face Emotion Detection using OpenCV

This repository demonstrates a **real-time face emotion detection system** using:
- Classical computer vision (OpenCV)
- A pre-trained deep learning model

The project bridges **computer vision and machine learning**, making it ideal for learning how AI systems interpret human facial expressions.

---

## Repository Structure

```text
Face_Emotion_detection_OpenCV_29062025-main/
├── face_emotion_detector.py
├── fer2013_mini_XCEPTION.102-0.66.hdf5
├── how_to_use.txt
└── .gitignore
```

---

## What This Project Does

The system:
1. Captures video from a camera
2. Detects faces in each frame
3. Extracts facial regions
4. Predicts emotional states
5. Displays emotion labels in real time

Typical detected emotions include:
- Happy
- Sad
- Angry
- Surprised
- Neutral

---

## Core Pipeline Breakdown

### 1. Video Capture

Frames are captured continuously from the webcam.

Analogy:  
A video stream is a **rapid slideshow** of images.

---

### 2. Face Detection

OpenCV Haar cascades or similar detectors locate faces.

Mathematically, detection works by:
- Sliding windows
- Feature comparison
- Threshold classification

Analogy:  
Like scanning a crowd for **face-shaped patterns**.

---

### 3. Preprocessing

Detected face regions are:
- Converted to grayscale
- Resized
- Normalized

Normalization equation:

```
x_norm = x / 255
```

This scales pixel values into `[0,1]`.

---

### 4. Emotion Classification (CNN)

A **Convolutional Neural Network** processes the face image.

Key operation (convolution):

```
Feature = Σ (Image × Kernel)
```

This extracts patterns such as:
- Eyebrows
- Mouth curvature
- Eye openness

Analogy:  
Like using multiple stencils to highlight facial features.

---

### 5. Prediction Output

The model outputs probabilities:

```
Emotion = argmax(P_emotion)
```

The emotion with the highest confidence is displayed.

---

## Using the Pre-Trained Model

- The `.hdf5` file contains learned weights
- Trained on the **FER-2013 dataset**
- No training required to run detection

This allows focus on **application and understanding**, not training.

---

## How to Run

1. Install dependencies:
```bash
pip install opencv-python tensorflow keras numpy
```

2. Run the script:
```bash
python face_emotion_detector.py
```

3. Ensure:
- Webcam is connected
- Model file is in the same directory

---

## Using This Repository as Notes

### Study Suggestions
- Pause after each pipeline step
- Print intermediate images
- Modify input resolution
- Replace webcam with video file

### Questions to Explore
- How lighting affects predictions
- Why grayscale is sufficient
- Model bias and limitations

---

## Requirements

- Python 3.x
- OpenCV
- TensorFlow / Keras
- NumPy

---

## Learning Outcomes

You will understand:
- Face detection fundamentals
- CNN-based classification
- Real-time vision pipelines
- Practical AI deployment

---

## Final Note

This project shows how **human emotion becomes numbers**, and how machines interpret those numbers to infer meaning.
