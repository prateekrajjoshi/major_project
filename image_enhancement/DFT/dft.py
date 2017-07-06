#using opencv FFT
import cv2
import numpy as np
from matplotlib import pyplot as plt


def makeGaussian(size, fwhm = 3, center=None):
    """ Make a square gaussian kernel.
    size is the length of a side of the square
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.
    """
    #fwhm basically controlls the width of the gaussian window	
    x = np.arange(0, size[0], 1, float)
    y = np.arange(0, size[1], 1, float)
    y = y[:,np.newaxis]
	

    if center is None:
        x0 = size[0] // 2 #removing decimal after division
	y0 = size[1] // 2
    else:
        x0 = center[0]
        y0 = center[1]
    
    return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)



img = cv2.imread('cutter_hist_den.png',0)


#fft transform
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20*np.log(np.abs(fshift))

#creation of LPF
rows = magnitude_spectrum.shape[0]
cols = magnitude_spectrum.shape[1]

#mask gaussian
mask = np.zeros((cols,rows),np.float32)
gaussian_sub  = makeGaussian((cols,rows),70)#last parameter is the width of gaussian window. smaller window, only few low freq components
mask = gaussian_sub

#to make rectangular window low pass filter
'''
mask = np.ones((cols,rows),np.float32)
mask[rows/2-50:rows/2+50,cols/2-50:cols/2-50] = 0
'''

#to make rectangular window high pass filter
'''
mask = np.zeros((cols,rows),np.float32)
mask[rows/2-50:rows/2+50,cols/2-50:cols/2-50] = 1
'''
#these two will have ringing effect


plt.subplot(121),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])


#convolution in frequency domain
f = fshift * mask
f_ishift = np.fft.ifftshift(f)

plt.subplot(122),plt.imshow(np.abs(np.fft.ifft2(f_ishift)), cmap = 'gray')
#plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.show()

cv2.imwrite("result1.jpg",np.abs(np.fft.ifft2(f_ishift)))



