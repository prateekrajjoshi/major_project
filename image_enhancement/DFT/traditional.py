
import cv2
import numpy as np
import cmath
from matplotlib import pyplot as plt

#computationally expensive N^4
def compute_dft(inp_img):
	output = np.zeros(inp_img.shape,dtype=np.complex128)
	temp = np.complex128(0.0 + 0.0 * 1j)

	dim1 = np.float32(inp_img.shape[0])
	dim2 = np.float32(inp_img.shape[1])

	i = inp_img.shape[0] - 1
	p = inp_img.shape[1] - 1
	k = inp_img.shape[0] - 1
	l = inp_img.shape[1] - 1

	N = np.float32((i+1)*(p+1))
	while i >= 0:
		while p >= 0:
			temp = np.complex128(0.0 + 0.0 * 1j)
			while k >= 0:			
				while l >= 0:
					temp += inp_img[k][l] * cmath.exp(-2j * cmath.pi * ((p * l) / dim2 + (i * k) / dim1))			    	
					l-=1
				l = inp_img.shape[1] - 1
				k-=1
			print i,p
			k = inp_img.shape[0] - 1		     
			p-=1
			output[i][p] = temp

		p = inp_img.shape[1] - 1		
		i-=1			
	return output



img = cv2.imread("small_rose.jpg",0)
dft = compute_dft(img)
fshift = np.fft.fftshift(dft)
magnitude_spectrum = 20*np.log(np.abs(fshift)) 
plt.subplot(311),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(312),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()
plt.subplot(313),plt.imshow(ifft(fshift), cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()


