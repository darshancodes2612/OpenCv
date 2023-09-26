import cv2 
 
img = cv2.imread("watch.jpg")
cv2.imshow("watch_img",img)
cv2.waitKey(0)

grayscale = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
resize_stretch = cv2.resize(grayscale,(780,540),interpolation = cv2.INTER_LINEAR)
cv2.imshow("grayscale",grayscale)
cv2.waitKey(0)
cv2.destroyAllWindows()