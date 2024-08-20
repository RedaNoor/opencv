import cv2
import streamlit as st
import numpy as np
from PIL import Image

# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize the video capture
cap = cv2.VideoCapture(0)

st.title("Real-Time Face Detection")

# Create a placeholder for the video stream
video_placeholder = st.empty()

while True:
    # Read the frame from the video capture
    ret, frame = cap.read()

    if not ret:
        st.write("Failed to capture video")
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Convert the frame to RGB format
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert the frame to a PIL image
    img = Image.fromarray(frame_rgb)

    # Display the frame in the Streamlit app
    video_placeholder.image(img, caption="Real-Time Face Detection", use_column_width=True)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture
cap.release()
