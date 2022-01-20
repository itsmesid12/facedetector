# -*- coding: utf-8 -*-
"""
Created on Sun Jan  2 14:15:22 2022

@author: Sidharth
"""

import cv2

def opencam(d):
    cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
    if not cap.isOpened():
        raise IOError("Cannot open webcam")
    try:
        while d:
            ret1, frame1 = cap.read()
            frame1 = cv2.resize(frame1, None, fx=1.0, fy=1.0, interpolation=cv2.INTERSECT_FULL)
            faces = face_cascade.detectMultiScale(frame1,1.1,4)
            for(x,y,w,h) in faces:
                cv2.rectangle(frame1,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.imshow('Facerecog',frame1)
            cv2.waitKey(1)
    except KeyboardInterrupt:
        cap.release()
        cv2.destroyAllWindows()

opencam(1)

