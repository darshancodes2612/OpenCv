from flask import Flask, Response
import cv2
import numpy as np

app = Flask(__name__)

# Initialize Haarcascades face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_faces_haarcascade(img, scaleFactor=1.1, minNeighbors=5):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors)
    return faces

def generate_frames():
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Reducing frame resolution to speed up face detection
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        faces = detect_faces_haarcascade(small_frame)

        for (x, y, w, h) in faces:
            x, y, w, h = 2 * x, 2 * y, 2 * w, 2 * h  # Adjusting coordinates for original frame size
            face_roi = frame[y:y+h, x:x+w]
        
        # Encode the frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
