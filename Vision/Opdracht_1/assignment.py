import cv2
import numpy as np

# Load image in grayscale
image = cv2.imread('example6.png', cv2.IMREAD_GRAYSCALE)

gaussian_blur_kernel = np.array([
    [1, 2, 1],
    [2, 4, 2],
    [1, 2, 1]
], dtype=np.float32) / 16

vertical_kernel = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]], dtype=np.float32)*1.7

horizontal_kernel = np.array([[-1, -2, -1],
                    [ 0,  0,  0],
                    [ 1,  2,  1]], dtype=np.float32)

sharpen_kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])


# Randen detecteren 
edges = cv2.filter2D(image, -1, vertical_kernel)

 # Randen filteren
image = cv2.inRange(edges, 100, 255)  

# Hough Line Transform
lines = cv2.HoughLinesP(image, 1, np.pi/180, threshold=100, minLineLength=50, maxLineGap=10)

# Convert grayscale to BGR for colored lines
output = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

# Draw lines
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(output, (x1, y1), (x2, y2), (0, 255, 0), 2)




# Show result
cv2.imshow('Hough Lines', output)
cv2.waitKey(0)
cv2.destroyAllWindows()