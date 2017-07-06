import numpy as np
import cv2
import glob
import pickle

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('./calibration_images/*.jpg')

for fname in images:
     img = cv2.imread(fname)
     gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

     # Find the chess board corners
     ret, corners = cv2.findChessboardCorners(gray, (9,6),None)

     # If found, add object points, image points (after refining them)
     if ret == True:
         objpoints.append(objp)
         cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
         imgpoints.append(corners)
         # Draw and display the corners
         cv2.drawChessboardCorners(img, (9,6), corners,ret)
         cv2.imshow('img',img)
         cv2.waitKey(200)

print type(imgpoints)
print len(imgpoints)
print type(objpoints)
print len(objpoints)
print imgpoints
print objpoints
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)



print "saving parameters in file....\n"

file_obj = open('./calibration_params/info.txt', 'w')
file_obj.write("fx and fy in pixel units are:\n")
file_obj.write(str(mtx[0,0]))
file_obj.write("\n") 
file_obj.write(str(mtx[1,1]))

file_obj.write("\n\n")
file_obj.write("principle points used as image centers cx and cy are:\n")
file_obj.write(str(mtx[0,2]))
file_obj.write("\n") 
file_obj.write(str(mtx[1,2]))

file_obj.write("\n\n")
file_obj.write("distortion parameters (k1, k2, p1, p2, k3) are:\n")
file_obj.write("(")
file_obj.write(str(dist[0,0]))
file_obj.write(", ") 
file_obj.write(str(dist[0,1]))
file_obj.write(", ")
file_obj.write(str(dist[0,2]))
file_obj.write(", ") 
file_obj.write(str(dist[0,3]))
file_obj.write(", ") 
file_obj.write(str(dist[0,4]))
file_obj.write(")")
file_obj.close()

file_obj = open('./calibration_params/dist.txt', 'w')
np.save(file_obj, dist)
file_obj.close()

file_obj1 = open('./calibration_params/mtx.txt', 'w')
np.save(file_obj1, mtx)
file_obj1.close()

with open('./calibration_params/rvecs.txt', 'wb') as fp:
    pickle.dump(rvecs, fp)
fp.close()

with open('./calibration_params/tvecs.txt', 'wb') as fp:
    pickle.dump(tvecs, fp)
fp.close()

print "saving completed!!!"

cv2.destroyAllWindows()


