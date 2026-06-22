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
import sys
import time
import numpy as np
import csv
from datetime import datetime

try:
    from deepface import DeepFace
except ImportError:
    DeepFace = None
    print("[WARNING] DeepFace not installed. Emotion detection will be disabled.")

# --- Configuration ---
DATASET_PATH = 'dataset'
ATTENDANCE_FILE = 'Attendance.csv'
DETECT_EMOTIONS = True  # Built-in option to toggle emotion sensing

def load_images_from_dataset(path):
    images = []
    classNames = []
    
    if not os.path.exists(path):
        print(f"[ERROR] '{path}' directory not found! Please create it and add face images.")
        sys.exit(1)
        
    for file in os.listdir(path):
        img_path = f'{path}/{file}'
        img = cv2.imread(img_path)
        if img is not None:
            images.append(img)
            classNames.append(os.path.splitext(file)[0])
        else:
            print(f"[WARNING] Could not read image: {file}")
            
    print(f"[INFO] Loaded {len(classNames)} Student Images: {classNames}")
    return images, classNames


def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodes = face_recognition.face_encodings(img)
        if len(encodes) > 0:
            encodeList.append(encodes[0])
        else:
            print("[WARNING] No face found in one of the loaded images.")
    return encodeList


marked_today = set()

def load_todays_attendance():
    today = datetime.now().strftime('%Y-%m-%d')
    try:
        with open(ATTENDANCE_FILE, 'r') as f:
            for row in csv.reader(f):
                if len(row) > 1 and row[1].startswith(today):
                    marked_today.add(row[0])
    except FileNotFoundError:
        pass

def markAttendance(name):
    if name in marked_today:
        return False
        
    today = datetime.now().strftime('%Y-%m-%d')
    with open(ATTENDANCE_FILE, 'a', newline='') as f:
        now = datetime.now()
        dtString = now.strftime('%Y-%m-%d %H:%M:%S')
        f.write(f'{name},{dtString}\n')
        print(f"[SUCCESS] Attendance marked for {name} at {dtString}")
        marked_today.add(name)
        return True

def main():
    print("[INFO] Starting FaceSentinel AI...")
    load_todays_attendance()
    images, classNames = load_images_from_dataset(DATASET_PATH)
    
    if not images:
        print("[ERROR] No valid images found. Exiting.")
        return

    print("[INFO] Encoding faces...")
    encodeListKnown = findEncodings(images)
    print("[INFO] Encoding Complete!")

    cap = cv2.VideoCapture(0)
    pTime = 0
    recently_marked = {}

    while True:
        success, img = cap.read()
        if not success:
            print("[ERROR] Failed to open camera. Exiting.")
            break

        # Calculate FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
        pTime = cTime

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
                is_new = markAttendance(name)
                
                if is_new:
                    recently_marked[name] = time.time()

                # Determine visual status
                box_color = (255, 165, 0)  # Orange for already logged
                status_text = " [LOGGED]"
                
                if name in recently_marked and (time.time() - recently_marked[name]) < 4:
                    box_color = (0, 255, 0)  # Green for just marked
                    status_text = " [JUST MARKED!]"

                # Emotion Sensing
                emotion_text = ""
                if DETECT_EMOTIONS and DeepFace is not None:
                    try:
                        # Crop face for emotion analysis to speed up processing
                        face_crop = img[y1:y2, x1:x2]
                        if face_crop.size > 0:
                            analysis = DeepFace.analyze(face_crop, actions=['emotion'], enforce_detection=False, silent=True)
                            # DeepFace returns a list of dicts or a dict depending on version
                            if isinstance(analysis, list):
                                analysis = analysis[0]
                            dom_emotion = analysis.get('dominant_emotion', '')
                            if dom_emotion:
                                emotion_text = f" ({dom_emotion})"
                    except Exception as e:
                        # Ignore errors if face is too blurry or DeepFace fails
                        pass

                display_text = f"{name}{status_text}{emotion_text}"

                # Draw fancy bounding box and name
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
                
                # Draw Box
                cv2.rectangle(img, (x1, y1), (x2, y2), box_color, 2)
                # Draw filled rectangle for name background
                cv2.rectangle(img, (x1, y2 - 35), (x2 + (len(display_text)*11), y2), box_color, cv2.FILLED)
                cv2.putText(img, display_text, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 2)

        # Display FPS
        cv2.putText(img, f'FPS: {int(fps)}', (20, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 2)

        cv2.imshow('FaceSentinel AI - Press Enter to Exit', img)

        if cv2.waitKey(1) == 13:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
