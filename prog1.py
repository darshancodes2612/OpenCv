import cv2


# Load the image
img = cv2.imread('watch.jpg')
cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()