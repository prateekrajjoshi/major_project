# A bit of setup
import numpy as np
import matplotlib.pyplot as plt
# Plot ad hoc mnist instances
from keras.datasets import mnist
from keras.utils import np_utils
import cv2

# load (downloaded if needed) the MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

#the architecture is same as that in the MNIST network, but the inputs and output layer is different




# flatten 28*28 images to a 784 vector for each image
num_pixels = X_train.shape[1] * X_train.shape[2]
X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')






# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255



# one hot encode outputs
#y_train = np_utils.to_categorical(y_train).astype('uint8')
#y_test = np_utils.to_categorical(y_test).astype('uint8')


y_train = y_train.astype('uint8')
y_test = y_test.astype('uint8')
num_classes = 10




#architecture setup, 2 hidden layers
#1st hidden layer of 128 neurons
#2nd hidden layer of 64 neurons
#activation = ReLU
#loss =  cross entropy, easy backpropagation of error

X = X_train
y = y_train


D =  num_pixels#dimensionality
K = num_classes # number of classes

#change D and K for input and output layer

h = 128 # size of hidden layer
h1 = 64 #if the number of neurons in these two layers are increased, the result is great but doubtful.
W = np.load('./weights_biases/weight_layer1.txt')
b = np.load('./weights_biases/bias_layer1.txt')
W1 = np.load('./weights_biases/weight_layer2.txt')
b1 = np.load('./weights_biases/bias_layer2.txt')
W2 = np.load('./weights_biases/weight_layer3.txt')
b2 = np.load('./weights_biases/bias_layer3.txt')

# some hyperparameters
step_size = 1e-1 #"h"
reg = 1e-3 # regularization strength,lambda

# gradient descent loop
num_examples = X.shape[0]
for i in xrange(500):#10000
  
  # evaluate class scores, [N x K]
  hidden_layer1 = np.maximum(0, np.dot(X, W) + b) # note, ReLU activation, 300x100
  hidden_layer2 = np.maximum(0, np.dot(hidden_layer1, W1) + b1) # note, ReLU activation, 300x100
  scores = np.dot(hidden_layer2, W2) + b2 #in every row, prediction for classes are stored for each input
  
  # compute the class probabilities
  exp_scores = np.exp(scores)
  probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # [N x K] # row = row/sum_row
  
  # compute the loss: average cross-entropy loss and regularization
  corect_logprobs = -np.log(probs[range(num_examples),y])#taking row corresponding to right class and performing cross entropy, kada logic
  data_loss = np.sum(corect_logprobs)/num_examples#cross entropy loss ko formula bata ako
  reg_loss = 0.5*reg*np.sum(W*W)+0.5*reg*np.sum(W1*W1)+ 0.5*reg*np.sum(W2*W2)#punishing W by square, concept of regularization
  loss = data_loss + reg_loss
  if i % 20 == 0:
    print "iteration %d: loss %f" % (i, loss)
  
  # compute the gradient on scores,how??????
  dscores = probs
  dscores[range(num_examples),y] -= 1 #easy derivation
  dscores /= num_examples
  



 # backpropate the gradient to the parameters
  # first backprop into parameters W2 and b2
  dW2 = np.dot(hidden_layer2.T, dscores)
  db2 = np.sum(dscores, axis=0, keepdims=True)
  # next backprop into hidden layer
  dhidden2 = np.dot(dscores, W2.T)
  # backprop the ReLU non-linearity
  dhidden2[hidden_layer2 <= 0] = 0#kada
  # finally into W,b
  dW1 = np.dot(hidden_layer1.T, dhidden2)
  db1 = np.sum(dhidden2, axis=0, keepdims=True)
  
  dhidden1 = np.dot(dhidden2, W1.T)
  # backprop the ReLU non-linearity
  dhidden1[hidden_layer1 <= 0] = 0#kada
  
  dW = np.dot(X.T, dhidden1)
  db = np.sum(dhidden1, axis=0, keepdims=True)
  
  # add regularization gradient contribution
  dW2 += reg * W2
  dW1 += reg * W1
  dW += reg * W
  
  # perform a parameter update
  W += -step_size * dW
  b += -step_size * db
  W1 += -step_size * dW1
  b1 += -step_size * db1
  W2 += -step_size * dW2
  b2 += -step_size * db2

file_obj = open('./weights_biases_clone/weight_layer1.txt', 'w')
np.save(file_obj, W)
file_obj.close()

file_obj1 = open('./weights_biases_clone/weight_layer2.txt', 'w')
np.save(file_obj1, W1)
file_obj1.close()

file_obj = open('./weights_biases_clone/weight_layer3.txt', 'w')
np.save(file_obj, W2)
file_obj.close()

file_obj = open('./weights_biases_clone/bias_layer1.txt', 'w')
np.save(file_obj, b)
file_obj.close()

file_obj1 = open('./weights_biases_clone/bias_layer2.txt', 'w')
np.save(file_obj1, b1)
file_obj1.close()

file_obj = open('./weights_biases_clone/bias_layer3.txt', 'w')
np.save(file_obj, b2)
file_obj.close()

 # evaluate training set accuracy
hidden_layer1 = np.maximum(0, np.dot(X_test, W) + b)
hidden_layer2 = np.maximum(0, np.dot(hidden_layer1, W1) + b1)
scores = np.dot(hidden_layer2, W2) + b2
predicted_class = np.argmax(scores, axis=1)#finding the predicted class
print 'training accuracy: %.2f' % (np.mean(predicted_class == y_test))#mean of result
print predicted_class-y_test #more zeros means prediction =  actual class

#result is promising today
#tomorrow's work -> take input as MNIST data and see the result
