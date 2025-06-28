import cv2
import numpy as np
from keras.models import load_model

# Load the pre-trained emotion detection model
emotion_model = load_model('fer2013_mini_XCEPTION.102-0.66.hdf5', compile=False)

# Load OpenCV's Haar cascade for face detection
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Define emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Start webcam feed
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

        # Extract the face ROI
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (64, 64), interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray]) != 0:
            # Preprocess for model
            roi = roi_gray.astype('float') / 255.0
            roi = np.expand_dims(roi, axis=0)       # Add batch dimension
            roi = np.expand_dims(roi, axis=-1)      # Add channel dimension

            # Predict emotion
            prediction = emotion_model.predict(roi, verbose=0)[0]
            label = emotion_labels[np.argmax(prediction)]
            label_position = (x, y - 10)

            # Put label
            cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, 'No Face', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the resulting frame
    cv2.imshow('Emotion Detector', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release capture and close windows
cap.release()
cv2.destroyAllWindows()
