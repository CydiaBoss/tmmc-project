# @title
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

import cv2

import ipywidgets as widgets
from ipywidgets import interact, interact_manual

from IPython.display import display, Javascript, Image

from google.colab.output import eval_js
from base64 import b64decode, b64encode
import PIL
import io
import html
import time

def houghCircleDetector(path_to_img):
    img = cv2.imread(path_to_img)

    # grayscale image
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Erode a bit
    # kernel = np.ones((7, 7), np.uint8)
    # gray = cv2.erode(gray, kernel, iterations=1)

    # Create Circle Detector
    params = cv2.SimpleBlobDetector_Params()

    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.8

    detector = cv2.SimpleBlobDetector_create(params)

    # Detect blobs in the image
    keypoints = detector.detect(gray)
    sizes = []

    # Draw circles on the original image
    for keypoint in keypoints:
        x = int(keypoint.pt[0])
        y = int(keypoint.pt[1])
        r = int(keypoint.size / 2)
        sizes.append(keypoint.size)
        cv2.circle(img, (x, y), r, (0, 255, 0), 2)

    sizes = np.around(np.divide(sizes,5),decimals=0)*5

    max_size = max(sizes)
    min_size = min(sizes)
    counter = 0

    for keypoint in keypoints:
        x = int(keypoint.pt[0])
        y = int(keypoint.pt[1])
        r = int(keypoint.size / 2)
        pc = "Partially Covered"
        nc = "Not Covered"
        c = "Covered"
        if sizes[counter] == max_size:
            cv2.putText(img,pc, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        elif sizes[counter] == min_size:
            cv2.putText(img,nc, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.putText(img,c, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        counter +=1

    plt.figure(figsize=(20,10))
    plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB)),plt.title('Result',color='c')
    plt.show()
    return

for i in range(1,15):
    houghCircleDetector(f"White_{i}.jpg")