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
     sol_goz_bolgesi = np.array([(yuzisareti.part(36).x, yuzisareti.part(36).y),
                                (yuzisareti.part(37).x, yuzisareti.part(37).y),
                                (yuzisareti.part(38).x, yuzisareti.part(38).y),
                                (yuzisareti.part(39).x, yuzisareti.part(39).y),
                                (yuzisareti.part(40).x, yuzisareti.part(40).y),
                                (yuzisareti.part(41).x, yuzisareti.part(41).y)], np.int32)
     min_x = np.min(sol_goz_bolgesi[:, 0])
     max_x = np.max(sol_goz_bolgesi[:, 0])
     min_y = np.min(sol_goz_bolgesi[:, 1])
     max_y = np.max(sol_goz_bolgesi[:, 1])

     goz = frame[min_y: max_y, min_x: max_x]
     gray_goz = cv2.cvtColor(goz, cv2.COLOR_BGR2GRAY)
     goz = cv2.resize(goz, None, fx=5, fy=5)


     cv2.imshow("Goz", goz)
     cv2.imshow("Frame",frame)
     key = cv2.waitKey(1)
     if key == 'q':
      break

cap.release()
cv2.destroyAllWindows()