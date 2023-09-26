import cv2
import numpy as np

# Load YOLO model and classes
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = f.read().strip().split('\n')

# Initialize VideoCapture
cap = cv2.VideoCapture(0)  # 0 for the default camera, you can change it to a video file path if needed

# Capture the initial frame
ret, frame = cap.read()
initial_frame = frame.copy()

while True:
    ret, frame = cap.read()

    # Detect objects in the frame
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_names = net.getUnconnectedOutLayersNames()
    detections = net.forward(layer_names)

    # Process detections
    for detection in detections:
        for obj in detection:
            scores = obj[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > 0.5:
                center_x, center_y, width, height = map(int, obj[0:4] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]]))
                x, y = int(center_x - width / 2), int(center_y - height / 2)
                
                # Check if the object is the same as in the initial frame
                if np.array_equal(frame[y:y+height, x:x+width], initial_frame[y:y+height, x:x+width]):
                    color = (0, 255, 0)  # Green bounding box for the initial object
                else:
                    color = (0, 0, 255)  # Red bounding box for foreign objects
                
                cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2)
                cv2.putText(frame, classes[class_id], (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display the frame
    cv2.imshow("Object Detection", frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release VideoCapture and close windows
cap.release()
cv2.destroyAllWindows()
