import cv2
import numpy as np
import dlib
from math import hypot

cap=cv2.VideoCapture(0)
detector=dlib.get_frontal_face_detector()
predictor=dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
while True:
 _, frame = cap.read()
 gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
 yuzler = detector(gray)


 for yuz in yuzler:
     yuzisareti=predictor(gray,yuz)

     burun_bolgesi = np.array([(yuzisareti.part(27).x, yuzisareti.part(27).y),
                             (yuzisareti.part(28).x, yuzisareti.part(28).y),
                             (yuzisareti.part(29).x, yuzisareti.part(29).y),
                             (yuzisareti.part(30).x, yuzisareti.part(30).y),
                             (yuzisareti.part(31).x, yuzisareti.part(31).y),
                             (yuzisareti.part(32).x, yuzisareti.part(32).y),
                             (yuzisareti.part(33).x, yuzisareti.part(33).y),
                             (yuzisareti.part(34).x, yuzisareti.part(34).y),
                             (yuzisareti.part(35).x, yuzisareti.part(35).y)], np.int32)
     min_x = np.min(burun_bolgesi[:, 0])
     max_x = np.max(burun_bolgesi[:, 0])
     min_y = np.min(burun_bolgesi[:, 1])
     max_y = np.max(burun_bolgesi[:, 1])

     burun = frame[min_y: max_y, min_x: max_x]
     gray_burun = cv2.cvtColor(burun, cv2.COLOR_BGR2GRAY)
     burun = cv2.resize(burun, None, fx=5, fy=5)

     cv2.imshow("burun", burun)
     cv2.imshow("Frame",frame)



     key = cv2.waitKey(1)
     if key == 'q':
         break

cap.release()
cv2.destroyAllWindows()

