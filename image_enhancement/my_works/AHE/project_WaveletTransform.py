from numpy import *
import scipy
from scipy import misc
#import pywt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image

#from pywt import thresholding

#image = misc.ascent().astype(float32)
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])


#image = Image.open('nt_toolbox/data/hibiscus.bmp').convert('LA')
#image.save('greyscale.png')
image = mpimg.imread('watch.jpg')
image = rgb2gray(image)
#plt.imshow(image)
#plt.show()
print image.shape

noiseSigma = 16.0
image += random.normal(0, noiseSigma, size=(image.shape))
plt.imshow(image)
plt.show()
wavelet = pywt.Wavelet('haar')
print wavelet
levels  = int( floor( log2(image.shape[0]) ) )

WaveletCoeffs = np.array(pywt.wavedec2( image, wavelet, level=9))
#print WaveletCoeffs
threshold = noiseSigma*sqrt(2*log2(image.size))
NewWaveletCoeffs = map (lambda x: pywt.threshold(x,threshold,'soft'), WaveletCoeffs)
#print NewWaveletCoeffs
NewImage = pywt.waverec2( NewWaveletCoeffs, wavelet)

plt.imshow(NewImage)
plt.show()
#plt.imshow(WaveletCoeffs)
#plt.show()

#def denoise(data,wavelet,noiseSigma):
 #   levels = int(floor(log2(data.shape[0])))
  #  WC = pywt.wavedec2(data,wavelet,level=levels)
#threshold=noiseSigma*sqrt(2*log2(image.size))
#NWC = map(lambda x: pywt.threshold(x,threshold,'soft'), WC)
#plt.imshow(NWC)
#plt.show()

#    return pywt.waverec2( NWC, wavelet)

Denoised={}
for wlt in pywt.wavelist():
    print wlt
