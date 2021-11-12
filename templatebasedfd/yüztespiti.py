import cv2
import numpy as np
import dlib

cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    yuzler = detector(gray)
    for yuz in yuzler:
        x1 = yuz.left()
        y1= yuz.top()
        x2 = yuz.right()
        y2 = yuz.bottom()
        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

        yuzisareti = predictor(gray, yuz)

        for n in range(0,68):
           x = yuzisareti.part(n).x
           y = yuzisareti.part(n).y
           cv2.circle(frame, (x, y), 4, (255,0,0),-1)



    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break