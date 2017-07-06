import cv2
import numpy as np
import math

pi=3.14159

img = cv2.imread('ir.png',0)
ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
contours,hierarchy = cv2.findContours(th2,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
cv2.imshow('ir_image',img)
cv2.waitKey(0)

#center of the contours or estimated image points of the world coordinates
coord = []
coord1 = []
i=len(contours)-1
while i>=0:
	cnt = contours[i]
	M = cv2.moments(cnt)
	x = M['m10']/M['m00']
	y = M['m01']/M['m00']
	coord.append([int(x),int(y)])
	coord1.append([int(x*math.cos(pi/18)-y*math.sin(pi/18)),int(x*math.sin(pi/18)+y*math.cos(pi/18))])	
	i-=1

coord1 = sorted(coord1)

for ind, cont in enumerate(contours):
    elps = cv2.fitEllipse(cont)
    cv2.ellipse(th2,elps,(255,0,0),2)


cv2.imshow('ellipses',th2)
cv2.waitKey(0)


font = cv2.FONT_HERSHEY_SIMPLEX
i=0
dummy = cv2.imread("ir.png")
while i<len(contours):
	temp = coord[i]
	cv2.circle(dummy,(temp[0],temp[1]), 2, (0,0,255), -1)
	i+=1

cv2.imshow('with centre of mass',dummy)
cv2.waitKey(0)
cv2.destroyAllWindows()

i=0
while i<len(contours):
	temp = coord1[i]
	cv2.putText(dummy,str(i),(int(temp[0]*math.cos(-pi/18)-temp[1]*math.sin(-pi/18)),int(temp[0]*math.sin(-pi/18)+temp[1]*math.cos(-pi/18))), font,0.5,(255,255,0),1,cv2.CV_AA)
	i+=1

cv2.imshow('numbering',dummy)
cv2.waitKey(0)
cv2.destroyAllWindows()

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)






