
import numpy as np

#activation function and its derivative
def nonlin(x,deriv=False):

	if(deriv==True):
		return x*(1-x) #derivative of sigmoid is f(x)(1-f(x)),simple dervation

	return 1/(1+np.exp(-x))


#X is training input and y is training output, I made about 20 examples
X = np.array([[0.07,0.08],

[0.09,0.01],

[0.71,0.12],

[0.13,0.14],

[0.01,0.02],

[0.73,0.04],

[0.13,0.17],

[0.08,0.02],

[0.01,0.04],

[0.19,0.17],

[0.68,0.06],

[0.02,0.41],

[0.49,0.31],

[0.18,0.06],

[0.01,0.39],

[0.13,0.17],

[0.19,0.50],

[0.6,0.39],

[0.23,0.18],

[0.11,0.22],

[0.44,0.33],

[0.37,0.1],

[0.87,0.01],

[0.45,0.33],

[0.80,0.1],

[0.47,0.01],

[0.45,0.23],

[0.70,0.1],

[0.57,0.02],

[0.55,0.29],

[0.61,0.12]
])


y = np.array([[0.15],

[0.1],

[0.83],

[0.27],

[0.03],

[0.77],

[0.30],

[0.1],

[0.05],

[0.36],

[0.74],

[0.43],

[0.8],

[0.24],

[0.40],

[0.30],

[0.69],

[0.99],

[0.41],

[0.33],

[0.77],

[0.47],

[0.89],

[0.88],

[0.81],

[0.48],

[0.68],

[0.71],

[0.59],

[0.84],

[0.73]
])


np.random.seed(1)


# randomly initialize our weights with mean 0

syn0 = 2*np.random.random((2,4)) - 1 
syn1 = 2*np.random.random((4,3)) - 1 
syn2 = 2*np.random.random((3,1)) - 1



for j in xrange(600000):

	# Feed forward through layers 0, 1, and 2
	l0 = X
	l1 = nonlin(np.dot(l0,syn0))
	l2 = nonlin(np.dot(l1,syn1))
	l3 = nonlin(np.dot(l2,syn2))


	# how much did we miss the target value?

	l3_error = y - l3


	if (j% 10000) == 0:#checking error value for every 100000 iterations
		print "Error:" + str(np.mean(np.abs(l3_error)))


	# in what direction is the target value?
	# were we really sure? if so, don't change too much.


	l3_delta = l3_error*nonlin(l3,deriv=True)

	# how much did each l2 value contribute to the l3 error (according to the weights)?

	l2_error = l3_delta.dot(syn2.T)	

	l2_delta = l2_error*nonlin(l2,deriv=True)


	# how much did each l1 value contribute to the l2 error (according to the weights)?

	l1_error = l2_delta.dot(syn1.T)

	# in what direction is the target l1?
	# were we really sure? if so, don't change too much.

	l1_delta = l1_error * nonlin(l1,deriv=True)

	syn2 += l2.T.dot(l3_delta)
	syn1 += l1.T.dot(l2_delta)
	syn0 += l0.T.dot(l1_delta)

#return {'y0':y0, 'y1':y1 ,'y2':y2 }



#this is testing part

#test is input

test = np.array([[0.17,0.02],

[0.15,0.39],

[0.31,0.22]])
#simple feedforward
l0 = test
l1 = nonlin(np.dot(l0,syn0))
l2 = nonlin(np.dot(l1,syn1))
l3 = nonlin(np.dot(l2,syn2))
#l3 is output
print l3
