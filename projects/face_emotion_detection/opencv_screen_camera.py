import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import ImageGrab
import pyautogui
from tensorflow import keras
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
classifier = load_model('Facce_emotion.h5')
class_labels=['disgust','happiness','surprise','neutral']
Aim = int(input("if you want to detect faces using camera input 1 elif you wanna detect faces on screen input 2 : "))

if Aim==1:

    video_capture = cv2.VideoCapture(0)
    face_locations = []

    while True:
        # Grab a single frame of video
        ret, frame = video_capture.read()

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Find all the faces in the current frame of video
        face_locations = face_cascade.detectMultiScale(gray, 1.1, 4)

        # Display the results
        for (x, y, w, h) in face_locations:
        # Draw a box around the face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            roi_gray=gray[y:y+h,x:x+w]
            roi_gray=cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)

            if np.sum([roi_gray])!=0:
                roi=roi_gray.astype('float')/255.0
                roi=img_to_array(roi)
                roi=np.expand_dims(roi,axis=0)

                preds=classifier.predict(roi)[0]
                label=class_labels[preds.argmax()]
                label_position=(x,y)
                cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)
            else:
                cv2.putText(frame,'No Face Found',(20,20),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)

        # Display the resulting image
        cv2.imshow('img',frame)
        k = cv2.waitKey(30) & 0xff
        if k==27:
            break
    video_capture.release()


elif Aim==2:
    cv2.namedWindow("Recording", cv2.WINDOW_NORMAL)
    while True:
        # Grab a single frame of screen
        img = pyautogui.screenshot(region=(45,25, 600, 900))
        frame = np.array(img)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Find all the faces in the current frame of video
        face_locations = face_cascade.detectMultiScale(gray, 1.1, 4)

        # Display the results
        for (x, y, w, h) in face_locations:
        # Draw a box around the face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            roi_gray=gray[y:y+h,x:x+w]
            roi_gray=cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)

            if np.sum([roi_gray])!=0:
                roi=roi_gray.astype('float')/255.0
                roi=img_to_array(roi)
                roi=np.expand_dims(roi,axis=0)

                preds=classifier.predict(roi)[0]
                label=class_labels[preds.argmax()]
                label_position=(x,y)
                cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)
            else:
                cv2.putText(frame,'No Face Found',(20,20),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)

        cv2.imshow('Recording',frame)
        k = cv2.waitKey(30) & 0xff
        if k==27:
            break
    cv2.destroyAllWindows()
