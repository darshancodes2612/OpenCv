import cv2 
import numpy as np 
from matplotlib import pyplot as plt

img  = cv2.imread("watch.jpg")
rgb_img = cv2.cvtColor(img,cv2.IMREAD_GRAYSCALE)

plt.imshow(rgb_img)

plt.waitforbuttonpress()
plt.close('all')