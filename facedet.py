# -*- coding: utf-8 -*-
"""
Created on Sun Jan  2 14:15:22 2022

@author: Sriram Sidhartha R
"""
#incluing opencv
import cv2

def opencam(d):
    #open web camera and capturing video
    cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    
    #haar casecade is a classifier which helps in detection of objects in the images and video. 
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
    
    #checking camera is working or not
    if not cap.isOpened():
        raise IOError("Cannot open webcam")
    try:
        while d:
            #reading and returning video frame by frame
            ret1, frame1 = cap.read()
            
            #take a frame and resizing it with interpolaion as intersect_full.
            frame1 = cv2.resize(frame1, None, fx=1.0, fy=1.0, interpolation=cv2.INTERSECT_FULL)
            
            #The below commented line will set the frames into thermal camera like mode 
            #frame1 = cv2.applyColorMap(frame1,cv2.COLORMAP_RAINBOW)
            
            #Help in finding the face using detectmultiscale
            faces = face_cascade.detectMultiScale(frame1,1.1,4)
            
            #Highlighting the face with rectangle of color(0,255,0) that is green rectangle. Colours can be changed by altering these (0-255,0-255,0-255). 
            for(x,y,w,h) in faces:
                cv2.rectangle(frame1,(x,y),(x+w,y+h),(0,255,0),2)
            
            #showing the frame with detected face 
            cv2.imshow('Facedet',frame1)
            cv2.waitKey(1)
            
            #the below line of codes help in closing the while loop through clicking onto the 'X' in the top right corner
            if cv2.getWindowProperty('Facedet', cv2.WND_PROP_VISIBLE) <1:
                break
    except KeyboardInterrupt:
        #It can also be interrupted using Ctrl + C.
        cap.release()
        cv2.destroyAllWindows()

opencam(1)

