""" Drowsiness Detection System
The following program is the Drowsiness Detection System model trainer.
The script requires numpy, openCV, tensorflow, and matplotlib be installed within the Python Environment.
"""

import numpy as np
import cv2 as openCV
import tensorflow
import matplotlib.pyplot as plt
import os
import random as rnd
from tensorflow import keras
from tensorflow.keras import layers

# Reading images into an array for image data and labels.
def_img = 224
train_eyes = []
Classes = ["Closed", "Open"]
data = "Test_Images/"
for eyes in Classes:
    image_path = os.path.join(data, eyes)
    class_idx = Classes.index(eyes)
    for images in os.listdir(image_path):
        img_array= openCV.imread(os.path.join(image_path, images), openCV.IMREAD_GRAYSCALE)
        image_rgb = openCV.cvtColor(img_array, openCV.COLOR_GRAY2RGB)
        cor_img_array = openCV.resize(image_rgb, (def_img, def_img))
        train_eyes.append([cor_img_array, class_idx])

# To avoid overfitting of the model, we shuffle the data.
rnd.shuffle(train_eyes)

X = []
y = []
for features, label in train_eyes:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1, def_img, def_img, 3)

# Normalize
X = X / 255.0
Y = np.array(y)

# MobileNet was designed for mobile-first system. Faster than others.
# # Transfer Learning's greatest advantage.
# # Transfer model that is trained in MobileNet application of Keras.
model = tensorflow.keras.applications.mobilenet.MobileNet()
# # Layer index 0 of MobileNet matches the default image size to use to train the model.//
base_input = model.layers[0].input
base_output = model.layers[-4].output # Dropout layer
#
# # Need to flatten the dropout layer
flatten = layers.Flatten()(base_output)
final_output = layers.Dense(1)(flatten)
final_output = layers.Activation("sigmoid")(final_output) # Binary Classification, thus we use sigmoid
# #
# # # Can also use SVM or any other simple learning techniques instead of huge DL technique
model = keras.Model(inputs = base_input, outputs = final_output)
model.compile(loss = "binary_crossentropy", optimizer = "adam", metrics = ["accuracy"])
model.fit(X, Y, epochs = 10, validation_split = 0.1)
model.save("my_newest_model.h5")

# Testing the model.
img_array = openCV.imread("sad_woman.jpg")
backToRgb = openCV.cvtColor(img_array, openCV.COLOR_BGR2RGB)
new_array = openCV.resize(backToRgb, (224, 224))
X_input = np.array(new_array).reshape(1, 224, 224, 3)
print(X_input.shape)
X_input = X_input / 255.0
pred = model.predict(X_input)
print(pred)

# Testing on other images.
plt.imshow(openCV.cvtColor(img_array, openCV.COLOR_BGR2RGB))
faceCascade = openCV.CascadeClassifier(openCV.data.haarcascades + "haarcascade_frontalface_default.xml")
eyeCascade = openCV.CascadeClassifier(openCV.data.haarcascades + "haarcascade_eye.xml")
gray = openCV.cvtColor(img_array, openCV.COLOR_BGR2GRAY)
eyes = eyeCascade.detectMultiScale(gray, 1.1, 4)
for (x, y, w, h) in eyes:
    roiOnGray = gray[y: y + h, x: x + w]
    roiOnColor = img_array[y: y + h, x: x + w]
    eye_cascade = eyeCascade.detectMultiScale(roiOnColor)
    if len(eye_cascade) == 0:
        print("Eyes not detected")
    else:
        for (x, y, w, h) in eye_cascade:
            roiOnEyes = roiOnColor[y: y + h, x: x + w]
    openCV.rectangle(img_array, (x, y), (x + w, y + h), (0, 255, 0), 2)
plt.imshow(openCV.cvtColor(roiOnEyes, openCV.COLOR_BGR2RGB))
final_img = openCV.resize(roiOnEyes, (224, 224))
final_img = np.expand_dims(final_img, axis = 0)
final_img = final_img / 255.0
pred = model.predict(final_img)
print(pred)
plt.show()