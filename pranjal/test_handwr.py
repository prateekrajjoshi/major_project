# A bit of setup
import numpy as np
import matplotlib.pyplot as plt
import cv2


W = np.load('./weights_biases_clone/weight_layer1.txt')
b = np.load('./weights_biases_clone/bias_layer1.txt')
W1 = np.load('./weights_biases_clone/weight_layer2.txt')
b1 = np.load('./weights_biases_clone/bias_layer2.txt')
W2 = np.load('./weights_biases_clone/weight_layer3.txt')
b2 = np.load('./weights_biases_clone/bias_layer3.txt')

img = cv2.imread('five_img.jpg',0)
num_pixels = img.shape[0]*img.shape[1]
img_flat = img.reshape(1, num_pixels).astype('float32')
X_test = img_flat.astype('float32')
#X_test = (255.0-X_test)/255.0

X_test = (X_test)/255.0

 # evaluate training set accuracy
hidden_layer1 = np.maximum(0, np.dot(X_test, W) + b)

hidden_layer2 = np.maximum(0, np.dot(hidden_layer1, W1) + b1)

scores = np.dot(hidden_layer2, W2) + b2

predicted_class = np.argmax(scores, axis=1)#finding the predicted class
#print 'training accuracy: %.2f' % (np.mean(predicted_class == y_test))#mean of result

print predicted_class #more zeros means prediction =  actual class

#result is promising today
#tomorrow's work -> take input as MNIST data and see the result
