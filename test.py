import cv2
import numpy as np

# Load image
testjpg='./Training Images/White/White_14.jpg'
img = cv2.imread(testjpg)

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Create a SimpleBlobDetector object
detector = cv2.SimpleBlobDetector_create()

# Detect blobs in the image
keypoints = detector.detect(gray)

# Draw circles on the original image
for keypoint in keypoints:
    x = int(keypoint.pt[0])
    y = int(keypoint.pt[1])
    r = int(keypoint.size / 2)
    cv2.circle(img, (x, y), r, (0, 255, 0), 2)

# Display the image
cv2.imshow("Detected Circles", img)
cv2.waitKey(0)
cv2.destroyAllWindows()