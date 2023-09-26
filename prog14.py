import cv2
import numpy as np 

vc = cv2.VideoCapture(0)

while True:
    _,img = vc.read()
    cv2.imshow("face",img)
    if cv2.waitKey(1) & 0xFF == ord('q') :
        break
vc.release()
vc.destroyAllWindows()