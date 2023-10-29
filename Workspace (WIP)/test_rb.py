import function as fn
import numpy as np
import matplotlib.pyplot as plt
import cv2

whiteImgs=fn.load_images_from_folder('./Training Images/White/')

testjpg='./Training Images/White/White_6.jpg'

sampleImg=fn.edgeDetector('./Training Images/White/White_6.jpg', 'canny')

#print(sampleImg)

whiteCords=np.where(sampleImg==255)
zeroVals=np.where(whiteCords[0]==0)
print(zeroVals[0])
print(whiteCords[0])

np.delete(whiteCords[0],zeroVals)
np.delete(whiteCords[1],zeroVals)

print(whiteCords)

#xVals=whiteCords[0].pop(zeroVals[0])
#yVals=whiteCords[1].pop(zeroVals[0])
#print(xVals,yVals)

print(sampleImg[0][85])
print(sampleImg.size)

sampleImgCrop = np.delete(sampleImg, (0,200))
#print(sampleImgCrop)

drawCircImg=cv2.circle(sampleImg,(120,50),20,(255,0,0),1)
cv2.imshow("test", drawCircImg) 

cv2.waitKey(0)
cv2.destroyAllWindows() 
