# histogram of oriented gradients -> hog method 
# what if the face is turned in diff direction -> find the facial landmarks and make a centralised image from that 
# using an already trained network
# here we have SVM classifier 
import cv2
import numpy as np 
import face_recognition
import os
from datetime import datetime 
# library understands things as rgb but we get the image as bgr 

path = 'py/images'
image = []
names = []
myList = os.listdir(path) #getting list of images from the folder 

for item in myList: 
    curr_img = cv2.imread(f'{path}/{item}')
    """ This function reads an image from the specified file path and returns it as a NumPy array.
        f'{path}/{item}' -> This is an f-string (formatted string literal) in Python.
        It constructs a string dynamically by inserting the values of path and item into the string at runtime.
    """
    image.append(curr_img)
    names.append(os.path.splitext(item)[0]) # 0 -> will only give us the name and not extension along with it 

def encode_images(image):
    encodeList = []
    for img in image:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB) #converting it to RGB so that we can work on it 
        encode = face_recognition.face_encodings(img)[0] # encodings are on the 0th index of the returned list
        encodeList.append(encode)
    return encodeList

def attendance_marker(name):
    with open('py/attendacne.csv','r+') as f: # opening the file with both read and write mode 
        myDataList = f.readlines() # reading from the file
        nameList =[]
        for line in myDataList:
            entry = line.split(',') # spliting it based on comma 
            nameList.append(entry[0]) # first element of entry is the name so taking the name from the name,time
           #print(nameList)
        if name not in nameList: #checing if attendance is alrady marked or not 
           now = datetime.now()
           dateString = now.strftime('%H:%M:%S') # we only want the time, not the date 
           f.writelines(f'\n{name},{dateString}')

encodeList = encode_images(image)
print("tata, bye bye, khtm, hogaya")

cap = cv2.VideoCapture(0)

while True:
    success, img= cap.read()
    imgS = cv2.resize(img,(0,0),None, 0.25, 0.25) # reducing the size of the image, so preocesswill be completed faster 
    imgS = cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)
    # note we can have multiple faces in the feed too, so we need to find the face locations and send the locations in the encoding function as well
    facesInCurrFrame = face_recognition.face_locations(imgS)
    encodingsOfCurrFrame = face_recognition.face_encodings(imgS,facesInCurrFrame)
    x1,x2,y1,y2= 0,0,0,0
    for encoding, faceLocation in zip(encodingsOfCurrFrame, facesInCurrFrame):
        match = face_recognition.compare_faces(encodeList,encoding) # comparing the encodings of the known and the test
        face_dist = face_recognition.face_distance(encodeList,encoding) # the image is compared from all and we gest a list of distances 
        # now from this list we can get the one with the least distance and that will be a match
        matchInd = np.argmin(face_dist) #finding the index of the image with min face distance
        
        if match[matchInd]:
            name = names[matchInd].upper() # extracting the name 
            y1,x2,y2,x1 = faceLocation
            y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4 # remember we reduced the size of the image to process it, now we need to undo that to make proper BOB aroung the actual image, so we multiply th size by 4, since earlier we made the img 1/4 th of is size, by using 0.25
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name, (x1+6,y2-6),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
            attendance_marker(name)
    cv2.imshow("Image",img)
    cv2.waitKey(1)


 