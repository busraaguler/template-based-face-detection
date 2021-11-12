
import cv2
import numpy as np
import dlib
from math import hypot

cap=cv2.VideoCapture(0)
detector=dlib.get_frontal_face_detector()
predictor=dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

#def midpoint(p1 ,p2):
    #return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)
#font=cv2.FONT_HERSHEY_PLAIN
#def get_blinking_ratio(eye_points, facial_landmarks):
    #left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
    #right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
    #center_top = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
    #center_bottom = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))

    #hor_line = cv2.line(frame, left_point, right_point, (0, 255, 0), 2)
    #ver_line = cv2.line(frame, center_top, center_bottom, (0, 255, 0), 2)

    #hor_line_lenght = hypot((left_point[0] - right_point[0]), left_point[1] - right_point[1])
    #ver_line_lenght = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))

    #ratio = hor_line_lenght / ver_line_lenght
    #return ratio
while True:
    _, frame =cap.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    faces=detector(gray)
    for face in faces:
      # x,y= face.left(),face.top()
      #x1,y1=face.right(),face.bottom()
      # cv2.rectangle(frame,(x,y),(x1,y1),(255,0,0),2)

       landmarks=predictor(gray,face)
       #göz kırpmayı tanıma işlemi
       #left_eye_ratio = get_blinking_ratio([36, 37, 38, 39, 40, 41], landmarks )
       #right_eye_ratio=get_blinking_ratio([42, 43, 44, 45, 46, 47],landmarks)
       #blinking_ratio= (left_eye_ratio + right_eye_ratio) / 2

       #if blinking_ratio > 5.7:
         #cv2.putText(frame, " ", (50,150), font ,4, (255,0,0))


       #bakış tanıma işlemi
       left_eye_region = np.array([(landmarks.part(36).x, landmarks.part(36).y),
                                   (landmarks.part(37).x, landmarks.part(37).y),
                                   (landmarks.part(38).x, landmarks.part(38).y),
                                   (landmarks.part(39).x, landmarks.part(39).y),
                                   (landmarks.part(40).x, landmarks.part(40).y),
                                   (landmarks.part(41).x, landmarks.part(41).y)], np.int32)

       #cv2.polylines(frame, [left_eye_region],True, (0,0,255),2)

       min_x = np.min(left_eye_region[:, 0])
       max_x = np.max(left_eye_region[:, 0])
       min_y = np.min(left_eye_region[:, 1])
       max_y = np.max(left_eye_region[:, 1])

       nose_region = np.array([(landmarks.part(27).x,landmarks.part(27).y),
                               (landmarks.part(28).x,landmarks.part(28).y),
                               (landmarks.part(29).x,landmarks.part(29).y),
                               (landmarks.part(30).x,landmarks.part(30).y),
                               (landmarks.part(31).x,landmarks.part(31).y),
                               (landmarks.part(32).x,landmarks.part(32).y),
                               (landmarks.part(33).x,landmarks.part(33).y),
                               (landmarks.part(34).x,landmarks.part(34).y),
                               (landmarks.part(35).x,landmarks.part(35).y)],np.int32)
       min_x = np.min(nose_region[:, 0])
       max_x = np.max(nose_region[:, 0])
       min_y = np.min(nose_region[:, 1])
       max_y = np.max(nose_region[:, 1])

   # for mounth_region in range (48,67):
      #  x = landmarks.part(mounth_region).x
      #  y = landmarks.part(mounth_region).y
       mounth_region = np.array([(landmarks.part(48).x, landmarks.part(48).y),
                                 (landmarks.part(49).x, landmarks.part(49).y),
                                 (landmarks.part(50).x, landmarks.part(50).y),
                                 (landmarks.part(51).x, landmarks.part(51).y),
                                 (landmarks.part(52).x, landmarks.part(52).y),
                                 (landmarks.part(53).x, landmarks.part(53).y),
                                 (landmarks.part(54).x, landmarks.part(54).y),
                                 (landmarks.part(55).x, landmarks.part(55).y),
                                 (landmarks.part(56).x, landmarks.part(56).y),
                                 (landmarks.part(57).x, landmarks.part(57).y),
                                 (landmarks.part(58).x, landmarks.part(58).y),
                                 (landmarks.part(59).x, landmarks.part(59).y),
                                 (landmarks.part(60).x, landmarks.part(60).y),
                                 (landmarks.part(61).x, landmarks.part(61).y),
                                 (landmarks.part(62).x, landmarks.part(62).y),
                                 (landmarks.part(63).x, landmarks.part(63).y),
                                 (landmarks.part(64).x, landmarks.part(64).y),
                                 (landmarks.part(65).x, landmarks.part(65).y),
                                 (landmarks.part(66).x, landmarks.part(66).y),
                                 (landmarks.part(67).x, landmarks.part(67).y)], np.int32)


       min_x = np.min(mounth_region[:, 0])
       max_x = np.max(mounth_region[:, 0])
       min_y = np.min(mounth_region[:, 1])
       max_y = np.max(mounth_region[:, 1])

       eye = frame[min_y: max_y, min_x: max_x]
       nose= frame[min_y: max_y, min_x: max_x]
       mounth = frame[min_y: max_y, min_x: max_x]

       gray_eye = cv2.cvtColor(eye,cv2.COLOR_BGR2GRAY)
       gray_nose = cv2.cvtColor(nose, cv2.COLOR_BGR2GRAY)
       gray_mounth= cv2.cvtColor(mounth,cv2.COLOR_BGR2GRAY)

       #_, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)
       #_, threshold_nose = cv2.threshold(gray_nose, 70, 255, cv2.THRESH_BINARY)
       #_, threshold_mounth = cv2.threshold(gray_mounth, 70, 255, cv2.THRESH_BINARY)

       #threshold_eye = cv2.resize(threshold_eye, None, fx=5, fy=5)
       #threshold_nose = cv2.resize(threshold_nose, None, fx=5, fy=5)
       #threshold_mounth = cv2.resize(threshold_mounth, None, fx=5, fy=5)

       mounth = cv2.resize(mounth, None, fx=5, fy=5)
       eye = cv2.resize(eye, None, fx=5, fy=5)
       nose=cv2.resize(nose, None, fx=5, fy=5)


    cv2.imshow("Eye" ,eye)
    cv2.imshow("agız", mounth)
    cv2.imshow("burun", nose)

    #cv2.imshow("Threshold", threshold_eye)
    #cv2.imshow("Threshold", threshold_nose)
    #cv2.imshow("Threshold", threshold_mounth)

    #cv2.imshow("Frame",frame)

    key=cv2.waitKey(1)
    if key=='q':
     break

cap.release()
cv2.destroyAllWindows()
