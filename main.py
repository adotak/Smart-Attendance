# pip install opencv-python face-recognition numpy
# pip install cmake
# pip install dlib
# pip install face-recognition


#folder
# rootDirectory
##--->dataset
###----->YourName.png
##--->Attendance.csv
##--->main.py

import cv2
import face_recognition
import os
import numpy as np
import csv
from datetime import datetime

path = 'dataset'
images = []
classNames = []

for file in os.listdir(path):
    img = cv2.imread(f'{path}/{file}')
    images.append(img)
    classNames.append(os.path.splitext(file)[0])

print("Student Images Loaded:", classNames)


def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)
        if len(encode) > 0:
            encodeList.append(encode[0])
    return encodeList

encodeListKnown = findEncodings(images)
print("Encoding Complete")


def markAttendance(name):
    today = datetime.now().strftime('%Y-%m-%d')
    attendance_file = 'Attendance.csv'

    try:
        with open(attendance_file, 'r') as f:
            existing_data = list(csv.reader(f))
    except FileNotFoundError:
        existing_data = []

    names_today = [row[0] for row in existing_data if len(row) > 1 and row[1].startswith(today)]
    if name in names_today:
        return
        
    with open(attendance_file, 'a', newline='') as f:
        now = datetime.now()
        dtString = now.strftime('%Y-%m-%d %H:%M:%S')
        f.write(f'{name},{dtString}\n')
        print(f"Attendance marked for {name} at {dtString}")

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        print("Failed to open camera")
        break

    imgSmall = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgSmall = cv2.cvtColor(imgSmall, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgSmall)
    encodesCurFrame = face_recognition.face_encodings(imgSmall, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            markAttendance(name)


            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(img, name, (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

    cv2.imshow('Webcam - Press Enter to Exit', img)

    if cv2.waitKey(1) == 13:
        break

cap.release()
cv2.destroyAllWindows()
