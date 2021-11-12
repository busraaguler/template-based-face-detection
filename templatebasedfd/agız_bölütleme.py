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
     print(yuz)


     agız_bölgesi = np.array([(yuzisareti.part(48).x, yuzisareti.part(48).y),
                               (yuzisareti.part(49).x, yuzisareti.part(49).y),
                               (yuzisareti.part(50).x, yuzisareti.part(50).y),
                               (yuzisareti.part(51).x, yuzisareti.part(51).y),
                               (yuzisareti.part(52).x, yuzisareti.part(52).y),
                               (yuzisareti.part(53).x, yuzisareti.part(53).y),
                               (yuzisareti.part(54).x, yuzisareti.part(54).y),
                               (yuzisareti.part(55).x, yuzisareti.part(55).y),
                               (yuzisareti.part(56).x, yuzisareti.part(56).y),
                               (yuzisareti.part(57).x, yuzisareti.part(57).y),
                               (yuzisareti.part(58).x, yuzisareti.part(58).y),
                               (yuzisareti.part(59).x, yuzisareti.part(59).y),
                               (yuzisareti.part(60).x, yuzisareti.part(60).y),
                               (yuzisareti.part(61).x, yuzisareti.part(61).y),
                               (yuzisareti.part(62).x, yuzisareti.part(62).y),
                               (yuzisareti.part(63).x, yuzisareti.part(63).y),
                               (yuzisareti.part(64).x, yuzisareti.part(64).y),
                               (yuzisareti.part(65).x, yuzisareti.part(65).y),
                               (yuzisareti.part(66).x, yuzisareti.part(66).y),
                               (yuzisareti.part(67).x, yuzisareti.part(67).y)], np.int32)


     min_x = np.min(agız_bölgesi[:, 0])
     max_x = np.max(agız_bölgesi[:, 0])
     min_y = np.min(agız_bölgesi[:, 1])
     max_y = np.max(agız_bölgesi[:, 1])

     agız = frame[min_y: max_y, min_x: max_x]
     gray_agiz = cv2.cvtColor(agız, cv2.COLOR_BGR2GRAY)
     agız = cv2.resize(agız, None, fx=5, fy=5)

     cv2.imshow("agız", agız)
     cv2.imshow("Frame",frame)
     key = cv2.waitKey(1)
     if key == 'q':
         break

cap.release()
cv2.destroyAllWindows()


