import cv2
import numpy as np

img = cv2.imread("image.jpg")
grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# edge detection kernel
k = np.array([[-1,-1,-1],
              [-1, 8,-1],
              [-1,-1,-1]])

e = cv2.filter2D(grey, -1, k)
rangefilter = cv2.inRange(e, 50, 255)

# geel alleen op randen
glow = np.zeros_like(img)
glow[rangefilter==255] = [0,255,255]
image = cv2.addWeighted(img, .4, glow, 1, 0)



cv2.imshow("Neon", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
