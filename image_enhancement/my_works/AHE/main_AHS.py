import cv2
import numpy as np
import math
from numpy import float32, uint32


def ahs_transorm_image(img,numerator,nor_hist,beta_val):
	#for converting 3D array to 2D
	shape_img = list(img.shape)	
	shape_img[2] = 1 
	two_D = tuple(shape_img)
	temp = np.zeros(two_D)
		
	tx_img = np.zeros(shape=(shape_img[0],shape_img[1], 3))

	 
	for c in xrange(0, 3):
		temp = img[:,:,c]
		for (x,y), value in np.ndenumerate(temp):
			try:
				tx_img[x,y,c] = int((255*numerator[c,temp[x,y]]) / (numerator[c,temp[x,y]] + beta_val[c] * (1-numerator[c,temp[x,y]]-nor_hist[c,temp[x,y]])))	
			except:
				tx_img[x,y,c] = 0
	return tx_img
