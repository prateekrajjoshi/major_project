import os
import pickle 
import sklearn
import numpy as np
from sklearn import cross_validation, grid_search
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.externals import joblib
import time

#Custom files
import transfer
#import feat
#import pickle_load

def arrayLoader(FLAGS, image_lists, category):
    z = 0
    for x in image_lists.keys():
        for y in category:
            z += len(image_lists[x][y])
    rep = np.zeros((z,2048))
    label = np.zeros((z,len(image_lists.keys())))
    x1 = 0
    y1 = 0
    for y in category:
        r,s,f = transfer.get_random_cached_bottlenecks(None, image_lists, -1, y, FLAGS.bottleneck_dir, FLAGS.image_dir, None, None)
        r =  np.array(r)
        print r.shape
        s = np.array(s)
        rep[x1 : x1 + r.shape[0], 0 : r.shape[1]] = r
        label[y1 : y1 + s.shape[0], 0 : s.shape[1]] = s
        x1 = r.shape[0]
        y1 = s.shape[0]
        del r,s,f
    del x1,y1
    
    temp = np.append(rep,label,axis=1)
    np.random.shuffle(temp)
    #   print temp.shape
    rep = np.hsplit(temp , [2048,2050])[0]
    label = np.hsplit(temp , [2048,2050])[1]
    del temp
    return rep, label

class batcher:
  def __init__(self,name):
    self.X = np.load(name)['representations']
    self.y = np.load(name)['y']
    

  def get_batch(self):
    idx = np.random.choice(len(self.X),size=20,replace=False)
    x_batch = self.X[idx, :]
    y_batch = self.y[idx,]
    return x_batch, y_batch
    

"""
train_svm_classifer will train a SVM, saved the trained and SVM model and
report the classification performance

features: array of input features
labels: array of labels associated with the input features
model_output_path: path for storing the trained svm model
"""
# save 20% of data for performance evaluation
#X_train, X_test, y_train, y_test = cross_validation.train_test_split(features, labels, test_size=0.2)



####~==DEBUG_CODES==~
class runner:

  def __init__(self, iteration=10000):
    self.iter = iteration
    self.batch1 = batcher('data_batch_1.npz')
    self.batch2 = batcher('data_batch_2.npz')
    self.batch3 = batcher('data_batch_3.npz')
    self.batch4 = batcher('data_batch_4.npz')
    self.batch5 = batcher('data_batch_5.npz')
    #batchT = batcher('test_batch.npz')
    #batch_loader = pickle_load.Dataset(path_to_cifar)

    ###~==END_OF_DEBUG==~

    '''FLAGS = transfer.returnFLAGS()
    print(FLAGS.image_dir)
    with open(os.path.join(FLAGS.bottleneck_dir,"info.pkl"), 'rb') as f:
        image_lists = pickle.load(f)
    X_train, y_tr = arrayLoader(FLAGS, image_lists, ['training','validation'])
    X_test, y_te = arrayLoader(FLAGS, image_lists, ['testing'])

    #y_train is in one-hot encoded form for use with tensorflow... converting it to simpler form)
    y_train = np.argmax(y_tr, axis=1)
    y_test = np.argmax(y_te, axis=1)
    print X_train

    print X_train.shape
    print y_train'''

    #self.X_test = np.load('test_batch.npz')['representations']
    #self.y_test = np.load('test_batch.npz')['y']


    ####~==DEBUG_CODES==~

    #random data generation in spiral form
    #np.random.seed(seed=int(time.time()))
    #N = 100 # number of points per class
    self.D = 2048 # dimensionality
    self.K = 10 # number of classes
    #X = np.zeros((1932,D))
    #y = np.zeros(1932, dtype='uint8')

    #X = X_train
    #y = y_train
    '''for j in xrange(K):
      ix = range(N*j,N*(j+1))
      r = np.linspace(0.0,1,N) # radius
      t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
      X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
      y[ix] = j'''
    #print X
    #print y

    #architecture setup, 2 hidden layers
    #1st hidden layer of 128 neurons
    #2nd hidden layer of 64 neurons
    #activation = ReLU
    #loss =  cross entropy, easy backpropagation of error

    #Loader = pickle_load(path_to_cifar)

    #change D and K for input and output layer

    self.h = 128 # size of hidden layer
    self.h1 = 64
    #np.random.seed(seed=int(time.time()))
    if os.path.exists('./weights_biases_clone'):
      self.W = np.load('weights_biases_clone/weight_layer1.txt')
      self.b = np.load('weights_biases_clone/bias_layer1.txt')
      self.W1 = np.load('weights_biases_clone/weight_layer2.txt')
      self.b1 = np.load('weights_biases_clone/bias_layer2.txt')
      self.W2 = np.load('weights_biases_clone/weight_layer3.txt')
      self.b2 = np.load('weights_biases_clone/bias_layer3.txt')
      print "Weights and biases loaded!!"
    else:  
      self.W = 0.01 * np.random.randn(D,h)
      print self.W
      self.b = np.zeros((1,h))
      #np.random.seed(3)#optional
      self.W1 = 0.01 * np.random.randn(h,h1)
      self.b1 = np.zeros((1,h1))
      self.W2 = 0.01 * np.random.randn(h1,K)
      self.b2 = np.zeros((1,K))
      print "Weights and biases initialized!!"



    # some hyperparameters
    self.step_size = 1e-4 #"h"
    self.reg = 1e-3 # regularization strength,lambda
    self.choice = 0
    self.c_count = 0
    # gradient descent loop

  def runner_train(self):

    start = time.time()

    for i in xrange(self.iter):#10000

      if self.choice == 0:
        X,y = self.batch3.get_batch()
        batcher_obj = self.batch3
        #print "In 0"
      elif self.choice == 1:
        X,y = self.batch4.get_batch()
        batcher_obj = self.batch4
        #print "In 1"
        #print "In 2"
      elif self.choice == 2:
        X,y = self.batch5.get_batch()
        batcher_obj = self.batch5
        #print "In 2"
      elif self.choice == 3:
        X,y = self.batch1.get_batch()
        batcher_obj = self.batch1
        #print "In 3"
      elif self.choice == 4:
        X,y = self.batch2.get_batch()
        batcher_obj = self.batch3
        #print "In 4"
      #elif self.choice == 6:
      #  X,y = self.batch4.get_batch()

      num_examples = X.shape[0]
      # evaluate class scores, [N x K]
      hidden_layer1 = np.maximum(0, np.dot(X, self.W) + self.b) # note, ReLU activation, 300x100
      hidden_layer2 = np.maximum(0, np.dot(hidden_layer1, self.W1) + self.b1) # note, ReLU activation, 300x100
      scores = np.dot(hidden_layer2, self.W2) + self.b2 #in every row, prediction for classes are stored for each input
      
      # compute the class probabilities
      exp_scores = np.exp(scores)
      probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # [N x K] # row = row/sum_row
      
      # compute the loss: average cross-entropy loss and regularization
      corect_logprobs = -np.log(probs[range(num_examples),y]) #taking row corresponding to right class and performing cross entropy, kada logic
      data_loss = np.sum(corect_logprobs)/num_examples#cross entropy loss ko formula bata ako
      reg_loss = 0.5*self.reg*np.sum(self.W*self.W)+0.5*self.reg*np.sum(self.W1*self.W1)+ 0.5*self.reg*np.sum(self.W2*self.W2)#punishing W by square, concept of regularization
      loss = data_loss + reg_loss
      if i % 100 == 0:
        print "iteration %d: loss %f" % (i, loss)
        self.runner_validation(batcher_obj=batcher_obj)

        
      
      # compute the gradient on scores,how??????
      dscores = probs
      dscores[range(num_examples),y] -= 1 #easy derivation
      dscores /= num_examples
      



     # backpropate the gradient to the parameters
      # first backprop into parameters W2 and b2
      dW2 = np.dot(hidden_layer2.T, dscores)
      db2 = np.sum(dscores, axis=0, keepdims=True)
      # next backprop into hidden layer
      dhidden2 = np.dot(dscores, self.W2.T)
      # backprop the ReLU non-linearity
      dhidden2[hidden_layer2 <= 0] = 0#kada
      # finally into W,b
      dW1 = np.dot(hidden_layer1.T, dhidden2)
      db1 = np.sum(dhidden2, axis=0, keepdims=True)
      
      dhidden1 = np.dot(dhidden2, self.W1.T)
      # backprop the ReLU non-linearity
      dhidden1[hidden_layer1 <= 0] = 0#kada
      
      dW = np.dot(X.T, dhidden1)
      db = np.sum(dhidden1, axis=0, keepdims=True)
      
      # add regularization gradient contribution
      dW2 += self.reg * self.W2
      dW1 += self.reg * self.W1
      dW += self.reg * self.W
      
      # perform a parameter update
      self.W += -self.step_size * dW
      self.b += -self.step_size * db
      self.W1 += -self.step_size * dW1
      self.b1 += -self.step_size * db1
      self.W2 += -self.step_size * dW2
      self.b2 += -self.step_size * db2

      self.c_count += 1
      if (self.c_count%15) == 0:
        self.choice += 1
        self.choice %= 5

      if (i % 1000) == 0 and i != 0:
        print("{} Time for image :{} ".format(i,time.time() - start))
        start = time.time()
        if not os.path.exists('./weights_biases_clone'):
          os.mkdir('./weights_biases_clone')


        file_obj = open('./weights_biases_clone/weight_layer1.txt', 'w')
        np.save(file_obj, self.W)
        file_obj.close()

        file_obj1 = open('./weights_biases_clone/weight_layer2.txt', 'w')
        np.save(file_obj1, self.W1)
        file_obj1.close()

        file_obj = open('./weights_biases_clone/weight_layer3.txt', 'w')
        np.save(file_obj, self.W2)
        file_obj.close()

        file_obj = open('./weights_biases_clone/bias_layer1.txt', 'w')
        np.save(file_obj, self.b)
        file_obj.close()

        file_obj1 = open('./weights_biases_clone/bias_layer2.txt', 'w')
        np.save(file_obj1, self.b1)
        file_obj1.close()

        file_obj = open('./weights_biases_clone/bias_layer3.txt', 'w')
        np.save(file_obj, self.b2)
        file_obj.close()

        print "Weights and biases saved!"
    self.runner_validation(final=True)


  def runner_validation(self,final=False,batcher_obj=None):
    
    if not final:
      X_test,y_test = batcher_obj.get_batch()
    else:
      X_test = np.load('test_batch.npz')['representations']
      y_test = np.load('test_batch.npz')['y']

      print X_test.shape
      print y_test.shape
    # evaluate training set accuracy
    hidden_layer1 = np.maximum(0, np.dot(X_test, self.W) + self.b)
    hidden_layer2 = np.maximum(0, np.dot(hidden_layer1, self.W1) + self.b1)
    scores = np.dot(hidden_layer2, self.W2) + self.b2
    predicted_class = np.argmax(scores, axis=1)#finding the predicted class
    
    if not final:
      print 'training accuracy: %.2f' % (np.mean(predicted_class == y_test))#mean of result
    else:
      print 'testing accuracy: %.2f' % (np.mean(predicted_class == y_test))
      print predicted_class-y_test #more zeros means prediction =  actual class

    #result is promising today
    #tomorrow's work -> take input as MNIST data and see the result

    ###~==END_OF_DEBUG==~


if __name__ == "__main__":
  run = runner()
  #run.runner_train()  
  run.runner_validation(final=True)






"""
model_output_path = os.path.join('Model_SVM','SVM')
param = [    {
        "kernel": ["linear"],
        #"C": [1, 10, 100, 1000]
        "C": [1]

    },
    {
        "kernel": ["rbf"],
        #"C": [1, 10, 100, 1000],
        "C": [1],
        "gamma": [1e-2]
    }
]

# request probability estimation
svm = SVC()#probability=True)

# 10-fold cross validation, use 4 thread as each fold and each parameter set can be train in parallel
clf = grid_search.GridSearchCV(svm, param,
        cv=10, n_jobs=4, verbose=3)

clf.fit(X_train, y_train)

if os.path.exists(model_output_path):
    joblib.dump(clf.best_estimator_, model_output_path)
else:
    print("Cannot save trained svm model to {0}.".format(model_output_path))

print("\nBest parameters set:")
print(clf.best_params_)

y_predict=clf.predict(X_test)
print y_predict
print type(y_predict)
print y_test
print type(y_test)
#labels=sorted(list(set(labels)))
#print("\nConfusion matrix:")
#print("Labels: {0}\n".format(",".join(labels)))
#print(confusion_matrix(y_test, y_predict, labels=labels))

print("\nClassification report:")
print(classification_report(y_test, y_predict))

print("Accuracy = %g %%" % (np.sum(y_predict == y_test) / float(y_test.shape[0]) * 100))
 
"""
