import cv2
from cvzone.ClassificationModule import Classifier
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import pyttsx3
import time

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

offset = 20
imgsize = 300

folder = "Data\Y"
counter = 0

labels = ["1", "2", "3", "A", "B", "C", "F", "G"]
while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgsize, imgsize, 3), np.uint8) * 255
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        imgCropShape = imgCrop.shape

        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgsize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgsize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgsize-wCal)/2)
            imgWhite[:, wGap:wCal+wGap] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            print(prediction, index)

        else:
            k = imgsize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgsize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgsize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)

        Txt1 = cv2.putText(imgOutput, labels[index], [x, y - 26], cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 0, 255), 2)
        cv2.rectangle(imgOutput, (x - offset, y - offset),
                      (x + w + offset, y + h + offset), (255, 0, 255), 4)

        textspeech = pyttsx3.init()
        textspeech.say(labels[index])
        textspeech.runAndWait()

        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", imgOutput)
    key = cv2.waitKey(1)
    if key == (ord("s")):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(counter)

