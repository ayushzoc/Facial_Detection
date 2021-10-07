""" Drowsiness Detection System
The following program is the Drowsiness Detection System. User needs to connect a camera before starting the program.
The script requires numpy, imutils, openCV, tensorflow, and matplotlib be installed within the Python Environment.
"""

import numpy as np
from imutils.video import VideoStream
import cv2 as openCV
import tensorflow
import winsound
import argparse

"""
Gets the camera and starts it for live video stream to provide feed to the program.
"""
argparser = argparse.ArgumentParser()
argparser.add_argument("-w", "--webcam", type = int, default = 0, help = "index of webcam on system")
args = vars(argparser.parse_args())
vs = VideoStream(src = args["webcam"]).start()

"""
Loads the model that is already saved within the Python Environment.
"""
model = tensorflow.keras.models.load_model("my_newest_model.h5")

"""
Defines the frequency and duration of the alarm sound.
"""
frequency = 2500
duration = 2000

"""
Initializes region of interest for eyes of dimension 224x224x3.
Done to match the image dimension of the training data and the model.
"""
roiOnEyes = np.zeros((224, 224, 3))

"""
CascadeClassifier helps pinpoint the area of interest in an image.
faceCascade is used to detect the facial structure.
eyeCascade is used to detect the eyes inside the faceCascade.
Either can be used without the other.
"""
faceCascade = openCV.CascadeClassifier(openCV.data.haarcascades + "haarcascade_frontalface_default.xml")
eyeCascade = openCV.CascadeClassifier(openCV.data.haarcascades + "haarcascade_eye.xml")
count = 0

# Continuous Loop
while True:
    """
    Gets the video feed from the camera, changes the image to gray to match the training data.
    Crops out regions of interest: eyes, face.
    """
    frame = vs.read()
    bgrToGray = openCV.cvtColor(frame, openCV.COLOR_BGR2GRAY)
    eyes = eyeCascade.detectMultiScale(bgrToGray, 1.1, 4)
    faceCascades = faceCascade.detectMultiScale(bgrToGray, 1.1, 4)

    """
    For every image recevied from video feed, rectangle frames are placed around the face and eyes.
    The image changed to grayscale is reverted to color and regions of interest are detected.
    """
    for (x, y, w, h) in eyes:
        roiOnGray = bgrToGray[y: y + h, x: x + w]
        roiOnColor = frame[y: y + h, x: x + w]
        openCV.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        eye_cascade = eyeCascade.detectMultiScale(roiOnColor)
        for (x, y, w, h) in eye_cascade:
            roiOnEyes = roiOnColor[y: y + h, x: x + w]
    """
    For every cropped out specific images obtained, images are resized to match the dimension of the images of the trained model.
    The images are resized by division with 255.
    Prediction of the model on the image is calculated and outputs are provided based on the prediction.
    """
    result = openCV.resize(roiOnEyes, (224, 224))
    result = np.expand_dims(result, axis=0)
    result = result / 255.0
    pred = model.predict(result)
    if pred != 1:
        print("Closed")
    else:
        print("Open")

    if pred != 1:
        display = "Closed Eyes"
        count += 1
        openCV.putText(frame, display, (150, 150), openCV.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 2, openCV.LINE_4)
        if count > 15:
            openCV.putText(frame, "Alert!", (150, 150), openCV.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            winsound.Beep(frequency, duration)
            count = 0
    else:
        display = "Open Eyes"
        openCV.putText(frame, display, (150, 150), openCV.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 2, openCV.LINE_4)

    # Provides rectangle frame around the face.
    # Kept out of the primary loop to avoid any miscalculations.
    for (x, y, w, h) in faceCascades:
        openCV.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    openCV.imshow("Detecting Drowsiness...", frame)
    if openCV.waitKey(1) & 0xFF == ord('q'):
        break
vs.stop()
openCV.destroyAllWindows()

















