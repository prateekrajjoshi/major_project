import cv2
import numpy as np
import math
import sys
from numpy import float32, uint32
from main_AHS import ahs_transorm_image

try:
	first_arg = sys.argv[1]

except Exception as err:
	print "argument not provided"
	

def histogram(img):
	#for converting 3D array to 2D
	shape_img = list(img.shape)	
	shape_img[2] = 1 
	two_D = tuple(shape_img)
	temp = np.zeros(two_D)
	
	#histogram
	hist = np.zeros((3,256))
	nor_hist = np.zeros((3,256)) 
	for c in xrange(0, 3):
		temp = img[:,:,c]
		for (x,y), value in np.ndenumerate(temp):
			hist[c,temp[x,y]] += 1	
	return hist

def find_img_type(histo):
	type_rgb = np.zeros((3))
	sum_rgb = np.zeros((3,3))
	temp = np.zeros((1,256))
	#finding the count of pixels in the ranges separated by 85 and 170	
	for c in xrange(0, 3):
		temp = histo[c,:]
		
		for x, value in np.ndenumerate(temp):
			if (x[0]>=0 and x[0]<86):
				sum_rgb[c][0] += temp[x]
			if (x[0]>=86 and x[0]<171): 
				sum_rgb[c][1] += temp[x]
			if (x[0]>=171): 
				sum_rgb[c][2] += temp[x]

	#finding the value of beta 0.8,1.1 or 1.5
	for c in xrange(0, 3):
		if ((sum_rgb[c][0]>sum_rgb[c][1]) and (sum_rgb[c][0]>sum_rgb[c][2])):
			type_rgb[c] = 0.8
		elif ((sum_rgb[c][2]>sum_rgb[c][0]) and (sum_rgb[c][2]>sum_rgb[c][1])):
			type_rgb[c] = 1.5
		else:
			type_rgb[c] = 1.1			
	return type_rgb

def prob_cdf(nmlzd_hist):	
	cdf = np.zeros((3,256))
	temp = np.zeros((256)) 
	for c in xrange(0, 3):
		temp = nmlzd_hist[c,:]
		for x, value in np.ndenumerate(temp):
			if x[0]>0:
				cdf[c,x] = temp[x[0]-1] + cdf[c,x[0]-1]	
	return cdf	
	
try:
	img = cv2.imread(first_arg)#select image 
	#to check if code is working make img[:,:,0] = const and see the change
	actual = histogram(img)

	norm = histogram(img).astype(np.int32)
	beta_val = find_img_type(norm)
	print beta_val #prints beta value of B,G,R channel in order
	
	#normalizing the matrix i.e probability of each intensity value
	shape = list(img.shape)
	nor_hist = actual/(shape[0]*shape[1]) #normalized value -> probability
	
	#calculate cdf
	numerator = prob_cdf(nor_hist)
	#transform image
	final_img = ahs_transorm_image(img,numerator,nor_hist,beta_val)
	gray = np.dot(final_img[...,:3], [0.299, 0.587, 0.114])
	cv2.imwrite("ahe_cut.jpg",gray)
	
except Exception as error:
	print error

