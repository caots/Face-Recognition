import cv2
import numpy as np
from PIL import Image
import os
import sqlite3

#tranning hinh anh nhan dien vs Thu vien nhan dien khuon mat
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()

#read file tranning
recognizer.read('recognizer/trainningData.yml')

# get profile by id from datatbase
def getProfile(id):
    conn = sqlite3.connect('/Users/mac/FaceBase.db')
    query = "SELECT * FROM people WHERE ID="+str(id)
    cursor = conn.execute(query)

    profile = None
    for row in cursor:
        profile = row
    conn.close()
    return profile

#use webcam
cap = cv2.VideoCapture(0)

#set text style
fontface = cv2.FONT_HERSHEY_SIMPLEX

while(True):
    #comment the next line and make sure the image being read is names img when using imread
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # ket hop webcam vs face_cascade de nhan dien khuon mat
    faces = face_cascade.detectMultiScale(gray,1.3,5)

    for (x,y,w,h) in faces:
        #rectangle: hinh chu nhat
        cv2.rectangle(frame, (x,y),( x+w, y+h ), (0,255,0),2)
        roi_gray = gray[y:y+h, x:x+w] # cut anh xam de so sanh voi dataTrain
        #roi_color = frame[y:y+h, x:x+w]

        id, confidence = recognizer.predict(roi_gray) # du doan anh voi data exists
        if confidence < 40:
            profile= getProfile(id)
            if(profile!= None):
                cv2.putText(frame, ""+str(profile[1]), (x + 10, y + h + 30), fontface, 1, (0,255,0), 2)
            
        else:
            cv2.putText(frame, "Unknown", (x + 10, y + h + 30), fontface, 1, (0, 0, 255), 2);

    cv2.imshow('photograph', frame)
    if(cv2.waitKey(1) == ord('q')):
        break

cap.release()
cv2.destroyAllWindows()


#---- document---
#https://medium.com/@ankit.bhadoriya/face-recognition-using-open-cv-part-3-574a0766cd84

        

    


    




