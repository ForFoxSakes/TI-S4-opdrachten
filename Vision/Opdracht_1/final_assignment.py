import cv2
import numpy as np

img = cv2.imread("image.jpg")
grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# edge detection kernel
k = np.array([[-1,-1,-1],
              [-1, 8,-1],
              [-1,-1,-1]])

# filter the image to detect edges
# and apply a threshold to create a binary image
# where edges are highlighted
# and the rest is black and white

e = cv2.filter2D(grey, -1, k)
rangefilter = cv2.inRange(e, 50, 255)

# apply neon glow effect
# create a glow effect by adding a colored overlay
glow = np.zeros_like(img)
glow[rangefilter==255] = [0,255,255]
image = cv2.addWeighted(img, .4, glow, 1, 0)
cv2.imshow("Neon", image)
cv2.waitKey(0)



rows, cols = img.shape[:2]
map_x = np.zeros((rows, cols), dtype=np.float32)
map_y = np.zeros((rows, cols), dtype=np.float32)

# vul map_x en map_y met een een sinusgolf eroverheen
for i in range(rows):
    for j in range(cols):
        map_x[i, j] = j + 20 * np.sin(i / 30.0)
        map_y[i, j] = i

output = cv2.remap(img,map_x,map_y,interpolation=cv2.INTER_LINEAR)
cv2.imshow("Funny mirror",output)

# using opencv to find contours in the binary image
contours, _ = cv2.findContours(rangefilter, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(image, contours, -1, (0, 255, 0), 2)
cv2.imshow('Contour Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
