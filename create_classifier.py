import numpy as np
from PIL import Image
import os, cv2



# Method to train custom classifier to recognize face
def train_classifer(name):
    # Read all the images in custom data-set
    path = os.path.join(os.getcwd()+"/data/"+name+"/")

    faces = []
    ids = []
    labels = []
    pictures = {}


    # Store images in a numpy format and ids of the user on the same index in imageNp and id lists

    for root,dirs,files in os.walk(path):
            pictures = files


    for pic in pictures :

            imgpath = path+pic
            img = Image.open(imgpath).convert('L')
            imageNp = np.array(img, 'uint8')
            id = int(pic.split(name)[0])
            #names[name].append(id)
            faces.append(imageNp)
            ids.append(id)

    ids = np.array(ids)

    #Train and save classifier
    # Check the OpenCV version
    if cv2.__version__.startswith('3'):
        # For OpenCV 3.x
        clf = cv2.face.LBPHFaceRecognizer_create()
    else:
        # For OpenCV 4.x
        clf = cv2.face.LBPHFaceRecognizer_create()

    clf.train(faces, ids)
    clf.write("./data/classifiers/"+name+"_classifier.xml")

