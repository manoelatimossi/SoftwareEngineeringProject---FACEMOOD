import cv2
import os
import numpy as np

#Trainers
eigentrain = cv2.face.EigenFaceRecognizer_create(num_components=8) #phantom pictures
fishertrain = cv2.face.FisherFaceRecognizer_create()
lbphtrain = cv2.face.LBPHFaceRecognizer_create()



def getImageId(): #runs through the photos fold and search for the ID
    paths=[os.path.join('fotos', f)for f in os.listdir('fotos')]
    faces = []
    ids = []
    #this for goes through the images and split the ids saving them in an id array and the images in the image array
    for Imagepath in paths:
        FaceImage = cv2.cvtColor(cv2.imread(Imagepath), cv2.COLOR_BGR2GRAY)
        id = int(os.path.split(Imagepath)[-1].split('.')[1])
        #print(id)
        ids.append(id)
        faces.append(FaceImage)
    return np.array(ids), faces

ids, faces = getImageId()
#print(faces)

print("trainning....")
eigentrain.train(faces, ids) #supervised trainning
eigentrain.write('ClassifierEigen.yml')

fishertrain.train(faces,ids)
fishertrain.write('ClassifierFisher.yml') #supervised trainning

lbphtrain.train(faces,ids)
lbphtrain.write('Classifierlbph.yml') #supervised trainning

print("Trainning was sucessful")

