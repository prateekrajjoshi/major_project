import numpy as np
import cv2
import glob
import sys
import pickle
import matplotlib as plt

#argument parsing part here
try:
	first_arg = sys.argv[1]
	second_arg = sys.argv[2]

except:
	print "arguments mismatch"

#file reading part




dist = np.load('./calibration_params/dist.txt')
mtx = np.load('./calibration_params/mtx.txt')

print dist.shape

with open ('./calibration_params/rvecs.txt', 'rb') as fp:
    rvecs = pickle.load(fp)
fp.close()

with open ('./calibration_params/tvecs.txt', 'rb') as fp:
    tvecs = pickle.load(fp)
fp.close()

#dist = np.array([-0.13615181, 0.53005398, 0, 0, 0]) # no translation

img = cv2.imread(first_arg)
h,  w = img.shape[:2]
newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))#changed 1 to 0
#changing to zero reduces unwanted pixels at border

# undistort
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
cv2.imshow('img',dst)
cv2.waitKey(0)

# crop the image
x,y,w,h = roi
dst = dst[y:y+h, x:x+w]
cv2.imwrite(second_arg,dst)

cv2.destroyAllWindows()

