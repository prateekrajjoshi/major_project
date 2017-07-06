import numpy as np
import cv2
import glob
import sys
import pickle




dist = np.load('./calibration_params/dist.txt')
mtx = np.load('./calibration_params/mtx.txt')

with open ('./calibration_params/rvecs.txt', 'rb') as fp:
    rvecs = pickle.load(fp)
fp.close()

with open ('./calibration_params/tvecs.txt', 'rb') as fp:
    tvecs = pickle.load(fp)
fp.close()

p = 0
for i in tvecs:
	print i
	print '\n'
	p+=1
print p
