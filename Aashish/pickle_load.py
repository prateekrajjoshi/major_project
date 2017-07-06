#! /usr/bin/env python

import numpy as np
import cPickle as pickle
import os
import cv2
import img_proc

class Dataset:
  
  def __init__(self, root_path):
    self.x_train, self.y_train, self.y_hot, self.x_test, self.y_test, self.label_names = self.load_CIFAR10(root_path)
    self.data_count = 0
    #self.x_train = img_proc.pre_process(self.x_train)
    self.total_data = len(self.y_train)

  def isBatchEnd(self, batch_size):
    #check if batch limit reached i.e need for shuffle
    #if not (self.total_data - self.data_count) > batch_size:
    if (self.total_data - self.data_count) < batch_size:
      self.data_count = 0
      print "Count limit reached. 'self.data_count' is reset."
      return True
    else:
      return False

  def load_CIFAR_batch(self,filename, meta = False):
    """ load single batch of cifar """
    with open(filename, 'rb') as f:
      datadict = pickle.load(f)
      if not meta:
        X = datadict['data']/ 255.0
        #X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float") #reshaping the image matrix changing the axes 
        Y = np.array(Y)
        #print Y.shape
        return X, Y
      else: 
        return datadict

  def load_CIFAR10(self, ROOT):
    """ load all of cifar """
    xs = []
    ys = []
    for b in range(1,6):
      f = os.path.join(ROOT, 'data_batch_%d' % (b))
      X, Y = self.load_CIFAR_batch(f)
      xs.append(X)
      ys.append(Y)    
      del X, Y
    Xtr = np.concatenate(xs) #loads training image
    Ytr = np.concatenate(ys) #loads training labels
    print Xtr.shape
    print Ytr.shape
    del xs , ys
    meta_batches = self.load_CIFAR_batch(os.path.join(ROOT, 'batches.meta'), meta = True)
    labels = meta_batches[ b'label_names']
    names = [x.decode('utf-8') for x in labels]
    labels = names
    del meta_batches, names
    Xte, Yte = self.load_CIFAR_batch(os.path.join(ROOT, 'test_batch')) #loads test images and test labels
    Y = self.one_hot(Ytr)
    return Xtr, Ytr, Y, Xte, Yte, labels

  def one_hot(self, Y_data):
    Y = np.zeros((len(Y_data), 10))
    for y1,y2 in zip(range(0,len(Y_data)),Y_data):
      Y[y1][y2] = 1
    del y1,y2
    print Y.shape
    return Y   

  def getNextBatch(self, batch_size):
    idx = np.random.choice(len(self.x_train),size=batch_size,replace=False)
    x_batch = self.x_train[idx, :, :, :]
    y_batch = self.y_train[idx,]
    y_hot = self.y_hot[idx, :]
    '''if self.isBatchEnd(batch_size):
      package = zip(self.x_train, self.y_train, self.y_hot)
      np.random.shuffle(package)
      x_i ,y_i, yh_i = zip(*package)  
      #self.x_train = np.concatenate(x_i, axis=0).reshape((50000,32,32,3))
      self.x_train = np.concatenate(x_i, axis=0).reshape((50000,32,32,3))
      self.y_train = np.array(y_i)
      #self.y_hot = np.concatenate(yh_i,axis=0).reshape((50000,10))
      self.y_hot = np.concatenate(yh_i,axis=0).reshape((50000,10))
      del x_i,y_i,yh_i,package'''
    #print self.x_train.shape
    #print self.y_train.shape
    #print self.y_hot.shape


    #x_batch = self.x_train[self.data_count:self.data_count+batch_size]
    #print x_batch.shape
    #y_batch = self.y_train[self.data_count:self.data_count+batch_size]
    #y_hot = self.y_hot[self.data_count:self.data_count+batch_size]

    #print y_batch.shape
    #print y_hot.shape
    self.data_count += batch_size
    return x_batch, y_batch, y_hot
'''if __name__ == "__main
  path_to_cifar = '/home/aashish/Documents/cifar-10-batches-py'
  data = Dataset(path_to_cifar)
  #for x_do in range (1, 10):
    #x,y,yh = data.getNextBatch(10000)
    #print yh
  x,y,yh = data.getNextBatch(10000)
  print x
  print y
  print yh

 # x,y,yh = data.getNextBatch(40000)
 # print y
 # print yh'''

"""x,y,yh = data.getNextBatch(10000)
  print yh
  print len(yh)
  x,y,yh = data.getNextBatch(40000)"""
    



