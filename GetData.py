import numpy as np
import cv2
import sqlite3
import os
#------ Update record to databse -------
def insertOrUpdate(id, name):
    #connect to db
    conn = sqlite3.connect("/Users/mac/FaceBase.db")
    
    #check if ID already exists
    query = "SELECT * FROM people WHERE ID="+str(id)
    cursor = conn.execute(query)
    isRecordExist=0
    for row in cursor:
        isRecordExist=1
    if(isRecordExist==1):
        query="UPDATE people SET Name='"+str(name)+"' WHERE ID="+str(id)
    else:
        query="INSERT INTO people(ID, Name) VALUES("+str(id)+",'"+str(name)+"')"

    print(query)  
    conn.execute(query)
    conn.commit()
    conn.close()

#------- nhan dien khuon mat vs webcam -------

# load thu vien nhan dang khuon mat default OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(0)

# insert database
id = input("Enter your ID: ")
name = input('Enter your Name: ')

insertOrUpdate(id, name)

# lay du lieu tu camera
sampleNum = 0
while(True):
    ret, frame = cap.read() # frame: data camera
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # convert to gray pricture
    
    # ket hop face_cascade vs webcam de cho ra gia tri khuon mat
    faces = face_cascade.detectMultiScale(gray, 1.3 ,5) #image,scaleFactor,minNeighbor

    for(x,y ,w,h) in faces:
        # ve hinh vuong nhan dien khuon mat
        cv2.rectangle(frame, (x,y), (x + w,y + h),(0,255,0) ,2)

        if not os.path.exists('dataSet'):
            os.makedirs('dataSet')
        
        #tang id
        sampleNum +=1
        
        #save the captured face in the dataset folder
        cv2.imwrite('dataSet/User.'+str(id)+'.'+str(sampleNum)+'.jpg', gray[y:y+h,x:x+w])

    cv2.imshow('frame',frame)
    cv2.waitKey(1)
    # break if the sample number is morethan 20
    if sampleNum>100:
         cap.release()
         cv2.destroyAllWindows()
         break;

cap.release()
cv2.destroyAllWindows()





#----------------------------------------------
# document: https://www.bogotobogo.com/python/OpenCV_Python/python_opencv3_Image_Object_Detection_Face_Detection_Haar_Cascade_Classifiers.php

    

    
    

