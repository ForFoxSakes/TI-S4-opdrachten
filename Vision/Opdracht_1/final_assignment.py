import cv2
import numpy as np

img = cv2.imread("image.jpg")
grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# edge detection kernel

k = np.array([[-1,-1,-1],
              [-1, 8,-1],
              [-1,-1,-1]])

# Rangefilter om het grijs te maken en vervolgens de randen te detecteren

e = cv2.filter2D(grey, -1, k)
rangefilter = cv2.inRange(e, 50, 255)

# voeg glow effect toe

glow = np.zeros_like(img)
glow[rangefilter==255] = [0,255,255]
image = cv2.addWeighted(img, .4, glow, 1, 0)
cv2.imshow("Neon", image)
cv2.waitKey(0)

# remap voorbeeld gepakt 

rows, cols = img.shape[:2]
map_x = np.zeros((rows, cols), dtype=np.float32)
map_y = np.zeros((rows, cols), dtype=np.float32)

# vul map_x en map_y met een een sinusgolf eroverheen

for i in range(rows):
    for j in range(cols):
        map_x[i, j] = j + 20 * np.sin(i / 30.0)
        map_y[i, j] = i+ 20 * np.sin(i / 30.0)

output = cv2.remap(img,map_x,map_y,interpolation=cv2.INTER_LINEAR)
cv2.imshow("Funny mirror",output)
cv2.waitKey(0)
cv2.destroyAllWindows()


# vanuit openCV voorbeeld contour detectie om te vergelijken met eigen
contours, _ = cv2.findContours(rangefilter, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(image, contours, -1, (0, 255, 0), 2)
cv2.imshow('Contour Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
