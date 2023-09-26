
import cv2
import numpy as np

# Initialize Haarcascades face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_faces_haarcascade(img, scaleFactor=1.1, minNeighbors=5):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors)
    return faces

# Feature extraction and matching using ORB
orb = cv2.ORB_create()

def extract_features(img):
    keypoints, descriptors = orb.detectAndCompute(img, None)
    return keypoints, descriptors

def match_faces(img1, img2):
    kp1, desc1 = extract_features(img1)
    kp2, desc2 = extract_features(img2)
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(desc1, desc2)
    
    return len(matches)

def main():
    cap = cv2.VideoCapture(0)
    ret, initial_frame = cap.read()
    if not ret:
        print("Failed to capture initial frame.")
        return
    
    cv2.imshow("Initial Face", initial_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    frame_skip = 1  # We'll perform feature matching every 5 frames
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Reducing frame resolution to speed up face detection
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        faces = detect_faces_haarcascade(small_frame)
        
        for (x, y, w, h) in faces:
            x, y, w, h = 2*x, 2*y, 2*w, 2*h  # Adjusting coordinates for original frame size
            face_roi = frame[y:y+h, x:x+w]

            if frame_count % frame_skip == 0:
                matches = match_faces(initial_frame, face_roi)
                if matches > 20:
                    color = (0, 255, 0)  # green
                else:
                    color = (0, 0, 255)  # red
                    print('Copy Case')
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        
        frame_count += 1
        cv2.imshow("Optimized Face Verification", frame)
        if cv2.waitKey(0.5) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
