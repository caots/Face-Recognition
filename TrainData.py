import cv2
import numpy as np
import os
from PIL import Image

# tranning hinh anh nhan dien
recognizer = cv2.face.LBPHFaceRecognizer_create()
path= 'dataSet'

def getImageWithID(path):
    #get the path of all the files in the folder
    imagePaths=[os.path.join(path, f) for f in os.listdir(path)]
    #print(imagePaths) : list url
    faces=[]
    IDs=[]

    for imagePath in imagePaths:
        #loading the image and converting it to gray scale
        faceImg = Image.open(imagePath).convert('L')
        #converting the PIL image into numpy array
        faceNp = np.array(faceImg,'uint8')
        #print(faceNp) : list matrix pixel

        #split to get ID of the image
        ID=int(imagePath.split('/')[1].split('.')[1])
        #print(os.path.split(imagePath)) => [dataSet, url]

        #add to array
        faces.append(faceNp)
        IDs.append(ID)
    
        cv2.imshow("traning",faceNp)
        cv2.waitKey(10)
        
    return IDs, faces
        
Ids, faces = getImageWithID(path)
recognizer.train(faces,np.array(Ids))
if not os.path.exists('recognizer'):
    os.makedirs('recognizer')
recognizer.save('recognizer/trainningData.yml')
cv2.destroyAllWindows()

#------------------
#install opencv-contrib-python : cv2.face
