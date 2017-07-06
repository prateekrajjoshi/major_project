import cv2
import numpy as np
from matplotlib import pyplot as plt
import sys
import math

try:
	first_arg = sys.argv[1]

except Exception as err:
	print "argument not provided"
	

def normalized_histogram(img):
	#for converting 3D array to 2D
	shape_img = list(img.shape)	
	shape_img[2] = 1 
	two_D = tuple(shape_img)
	temp = np.zeros(two_D)
	
	#histogram
	hist = np.zeros((3,256))
	for c in xrange(0, 3):
		temp = img[:,:,c]
		for (x,y), value in np.ndenumerate(temp):
			hist[c,temp[x,y]] += 1
	
	#normalizing histogram --> computing probability
	hist /= (shape_img[0]*shape_img[1])
	return hist

def calc_entropy(nor):
	entr_rgb = np.zeros((3))
	temp = np.zeros((1,256))
	for c in xrange(0, 3):
		entr = 0.0
		temp = nor[c,:]
		for x, value in np.ndenumerate(temp):
			if (temp[x]>0): #removing zero so that log is not infinity
				entr += temp[x] * math.log(temp[x],2) #shannon formula
		entr_rgb[c] = -entr
	return entr_rgb
	
	
try:
	img = cv2.imread(first_arg)#select image 
	img[:,:,1] =0
	#to check if code is working make img[:,:,0] = const and see the change
	norm = normalized_histogram(img) 
	entropy = calc_entropy(norm)
	print entropy #prints entropy of B,G,R channel in order
	

except Exception as error:
	print error

