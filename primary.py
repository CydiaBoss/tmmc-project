import cv2
import numpy as np

testjpg='./Training Images/White/White_1.jpg'

# Load image
img = cv2.imread(testjpg)

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Create a SimpleBlobDetector object
detector = cv2.SimpleBlobDetector_create()

# Detect blobs in the image
keypoints = detector.detect(gray)
sizes = []

# Draw circles on the original image
for keypoint in keypoints:
    x = int(keypoint.pt[0])
    y = int(keypoint.pt[1])
    r = int(keypoint.size / 2)
    cv2.circle(img, (x, y), r, (0, 255, 0), 2)
    sizes.append(keypoint.size)

sizes=np.around(np.divide(sizes,5),decimals=0)*5

max=max(sizes)
min=min(sizes)
counter=0

for keypoint in keypoints:
    x = int(keypoint.pt[0])
    y = int(keypoint.pt[1])
    r = int(keypoint.size / 2)
    pc = "Partially Covered"
    nc = "Not Covered"
    c = "Covered"
    if sizes[counter] == max:
        cv2.putText(img,pc, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    elif sizes[counter] == min:
        cv2.putText(img,nc, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        cv2.putText(img,c, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    counter +=1

# Display the image
cv2.imshow("Detected Circles", img)


cv2.waitKey(0)
cv2.destroyAllWindows()