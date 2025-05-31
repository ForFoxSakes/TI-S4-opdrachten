import cv2
import numpy as np

image = cv2.imread("image.jpg")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

edge_detection_kernel = np.array([  [-1, -1, -1],
                                    [-1, 8, -1],
                                    [-1, -1, -1]])

sharpen_kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])

blur_kernel = np.array([[0.0625, 0.125, 0.0625],
                          [0.125, 0.25, 0.125],
                          [0.0625, 0.125, 0.0625]])

sharpened = cv2.filter2D(gray, -1, sharpen_kernel)
edge_detection =cv2.filter2D(sharpened, -1, edge_detection_kernel)
blur = cv2.filter2D(edge_detection, -1, blur_kernel)


# Display the original and processed images
#cv2.imshow("Sharpened, Edge Detection, Blur", np.concatenate((sharpened, edge_detection,blur), axis=1))

# Display horror filter
filtered_image =cv2.inRange(image, np.array([0, 0, 100]), np.array([100, 100, 255]))
cv2.imshow("Filtered Image", filtered_image)



cv2.waitKey(0)
cv2.destroyAllWindows()
