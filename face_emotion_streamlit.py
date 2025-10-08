import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
import cv2
from PIL import Image
import matplotlib.pyplot as plt

loaded_model = load_model("C:/Users/mukul/Downloads/models(complete)/FaceEmotion/best_model.keras")

st.title("Face Emotion Detector")
st.sidebar.title("History")

file_upload = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if file_upload is not None:
    image = Image.open(file_upload)
    image_np = np.array(image)
    image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)  
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) > 0:
        faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
        (x, y, w, h) = faces[0]

        face_img = gray[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (48, 48))
        face_img = face_img / 255.0
        face_img = np.expand_dims(face_img, axis=[0, -1])

        classes = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy',
                   4: 'Neutral', 5: 'Sad', 6: 'Surprise'}

        # Predict emotion
        pred = loaded_model.predict(face_img)
        prediction = np.argmax(pred)
        label = classes[prediction]

        # Draw rectangle and label on image
        cv2.rectangle(image_cv, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(image_cv, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 255, 0), 2)

        
        display_img = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)

        st.image(display_img, caption=f"Detected Emotion: {label}", use_column_width=True)
        st.success(f"Predicted Emotion: **{label}** ")

    else:
        st.warning("No face detected in the image. Please try another one.")









