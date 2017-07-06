import cv2
import numpy as np
from matplotlib import pyplot as plt
img = cv2.imread('rose.jpg')
img1 = cv2.imread('clahe_rose.jpg')
img2 = cv2.imread('hist_rose.jpg')
img3 = cv2.imread('watch.jpg')
img4 = cv2.imread('clahe_watch.jpg')

color = ('b','g','r')
'''
plt.figure(1)
plt.subplot(311)
plt.hist(img.ravel(),256,[0,256]);

plt.subplot(312)
plt.hist(img1.ravel(),256,[0,256]);

plt.figure(3)
#plt.subplot(313)
plt.hist(img2.ravel(),256,[0,256]);
'''

for i,col in enumerate(color):
    histr = cv2.calcHist([img],[i],None,[256],[0,256])  
    plt.figure(1)
    plt.title("main image")
    plt.plot(histr,color = col)
    plt.xlim([0,256])
	
for i,col in enumerate(color):
    histr = cv2.calcHist([img1],[i],None,[256],[0,256])
    plt.figure(2) 
    plt.title("ACE image") 
    plt.plot(histr,color = col)
    plt.xlim([0,256])

for i,col in enumerate(color):
    histr = cv2.calcHist([img2],[i],None,[256],[0,256])  
    plt.figure(3)
    plt.title("AHE image")
    plt.plot(histr,color = col)
    plt.xlim([0,256])

for i,col in enumerate(color):
    histr = cv2.calcHist([img3],[i],None,[256],[0,256])
    plt.figure(4)  
    plt.title("hist_eq image")
    plt.plot(histr,color = col)
    plt.xlim([0,256])

for i,col in enumerate(color):
    histr = cv2.calcHist([img4],[i],None,[256],[0,256])  
    plt.figure(5)
    plt.title("CLAHE image")
    plt.plot(histr,color = col)
    plt.xlim([0,256])

plt.show()
