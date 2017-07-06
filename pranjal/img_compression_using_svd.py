#uses top eigen vectors to represent image also called PCA(principle component analysis)

#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
import time

from PIL import Image
img = Image.open('apple.jpg')
imggray = img.convert('LA') #original image pointer is imggray

#converting image to numpy matrix
imgmat = np.array(list(imggray.getdata(band=0)), float) 
imgmat.shape = (imggray.size[1], imggray.size[0]) 
imgmat = np.matrix(imgmat) #matrix representing image

#print imgmat.shape --> 128x127

U, sigma, V = np.linalg.svd(imgmat)

for i in xrange(1, 51, 5):
    reconstimg = np.matrix(U[:, :i]) * np.diag(sigma[:i]) * np.matrix(V[:i, :])
    plt.imshow(reconstimg, cmap='gray')
    title = "n = %s" % i
    plt.title(title)
    plt.show()

