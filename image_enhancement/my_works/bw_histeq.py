import cv2
import numpy as np
from matplotlib import pyplot as plt
import sys

try:
	first_arg = sys.argv[1]
	second_arg = sys.argv[2]
	third_arg = sys.argv[3]

except Exception as err:
	print "arguments not sufficient"
	print "format:"
	print "python hist_eq.py source dest1 dest2\n"	

def equalize_hist(img):
       img = cv2.equalizeHist(img) #histogram equalization for RGB channels separately
       return img

def CLAHE_equalize_hist(img):
      clahe = cv2.createCLAHE()	
      img = clahe.apply(img) 
      return img

try:
	img = cv2.imread(first_arg,0)#select image
	equ = equalize_hist(img) 
	cv2.imwrite(second_arg,equ)#target image

	img_for_clahe = cv2.imread(first_arg,0)#select image
	res = CLAHE_equalize_hist(img_for_clahe) 
	cv2.imwrite(third_arg,res)#target image

except Exception as error:
	print error

