#!/usr/bin/env python
 
import numpy as np
import cv2
import sys
 
def build_filters():
	filters = []
 	ksize = 31
 	for theta in np.arange(0, np.pi, np.pi/16):
 		kern = cv2.getGaborKernel((ksize, ksize), 4.0, theta, 10, 0.5, 0, ktype=cv2.CV_32F)#cv2.getGaborKernel(ksize, sigma, theta, lambda, gamma, psi, ktype=cv2.CV_32F)
 		kern /= 1.5*kern.sum()#change lamda and see the results
 		filters.append(kern)
 	return filters
 
def process(img, filters):
 	accum = np.zeros_like(img)
 	for kern in filters:
 		fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
 		np.maximum(accum, fimg, accum)
 	return accum
 


try:
	 img_fn = sys.argv[1]
	 img_dest = sys.argv[2]
except:
 	img_fn = 'test.png'
 
img = cv2.imread(img_fn,0)

if img is None:
	 print 'Failed to load image file:', img_fn
 	 sys.exit(1)
 
filters = build_filters()
res1 = process(img, filters)
res2 = np.maximum(img,res1)
cv2.imshow('result', res1)
cv2.waitKey(0)
cv2.imshow(img_dest, res2)
cv2.waitKey(0)
cv2.imwrite(img_dest,res2)

#img = cv2.cvtColor(img_fn,cv2.COLOR_BGR2GRAY)

#dsto = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
cv2.imshow("opencv method", dsto)
cv2.waitKey(0)
cv2.destroyAllWindows()
