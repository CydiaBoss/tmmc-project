import function as fn
import numpy as np
import matplotlib.pyplot as plt
import cv2

def CircleDetector(path_to_img : str):
    img = cv2.imread(path_to_img)
    img = cv2.medianBlur(img,3)
    img_edge = cv2.Canny(img,100,200)

    circles = cv2.HoughCircles(img_edge,cv2.HOUGH_GRADIENT,1,minDist=50,param1=200,param2=31, maxRadius=1000)
    return circles

#total_circles = fn.houghCircleDetector('./Training Images/White/White_6.jpg')


#whiteImgs=fn.load_images_from_folder('./Training Images/White/')

#print(whiteImgs[1])

testjpg='./Training Images/White/White_9.jpg'
#testjpg='./Training Images/White/6_some_holes_covered_1_partial_with_light.jpg'

verti_line_mat = np.float32([[-1,2,-1],
                            [-1,2,-1],
                            [-1,2,-1]])

MainImgBGR = cv2.imread(testjpg,cv2.IMREAD_UNCHANGED)
img_hsv=cv2.GaussianBlur(MainImgBGR, (7, 7), 0) 
img_hsv = cv2.cvtColor(img_hsv,cv2.COLOR_BGR2GRAY)
img_hsv = cv2.convertScaleAbs(img_hsv,2,10)
plt.imshow(img_hsv)
plt.savefig('test.jpg')
plt.show()

test = fn.houghCircleDetector('./test.jpg')

#sampleImg=fn.houghCircleDetector('./Training Images/White/White_6.jpg')

#sampleImg=fn.edgeDetector('./Training Images/White/White_6.jpg', 'canny')
#sampleImg=fn.lineDetector('./Training Images/White/White_6.jpg', verti_line_mat)
#print(sampleImg)

#print(plt.imread(testjpg))
#print(sampleImg)

