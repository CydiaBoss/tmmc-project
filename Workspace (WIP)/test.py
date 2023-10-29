import numpy as np
import cv2

testjpg='./Training Images/White/White_14.jpg'

image = cv2.imread(testjpg)
original = image.copy()
image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
lower = np.array([0, 0, 0], dtype="uint8")
upper = np.array([180, 255, 40], dtype="uint8")
mask = cv2.inRange(image, lower, upper)
mask = 255 - mask
output = cv2.bitwise_and(image, image, mask = mask)

# Find contours
cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# Extract contours depending on OpenCV version
cnts = cnts[0] if len(cnts) == 2 else cnts[1]

# Iterate through contours and filter by the number of vertices 
for c in cnts:
    perimeter = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.04 * perimeter, True)
    if len(approx) > 5:
        cv2.drawContours(original, [c], -1, (36, 255, 12), -1)

cv2.imshow('mask', mask)
cv2.imshow('original', original)
cv2.imshow('output', output)
cv2.imwrite('mask.png', mask)
cv2.imwrite('original.png', original)
cv2.imshow('output.png', output)
cv2.waitKey()