import numpy as np

def nonlin(x,deriv=False):
	to_return = np.zeros_like(x)
	
	#for derivative
	if(deriv==True):
		to_return[x > 0] = 1	
		return to_return		
					
	#for activation
	np.maximum(to_return, x, to_return)			
	return to_return



X = np.array([[0.07,0.08],

[-0.09,0.01],

[0.71,-0.12],

[0.13,0.14]])

print X
print "======================"
print nonlin(X,True)	