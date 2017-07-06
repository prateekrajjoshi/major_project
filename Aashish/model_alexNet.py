import tensorflow as tf

# Create AlexNet model

class modelAlexNet:

    def __init__(self):
        #self.prediction = self.alex_net(self,_X, _dropout, n_classes, imagesize, img_channel)
        pass

    def model_predict(self,_X,_dropout, n_classes, imagesize, img_channel):
        self.prediction = self.alex_net(_X, _dropout, n_classes, imagesize, img_channel)
        return self.prediction

    def conv1st(self,name, l_input, w, b):
        cov = tf.nn.conv2d(l_input, w, strides=[1, 1, 1, 1], padding='SAME')
        return tf.nn.relu(tf.nn.bias_add(cov,b), name=name)
        
    def conv2d(self,name, l_input, w, b):
        cov = tf.nn.conv2d(l_input, w, strides=[1, 1, 1, 1], padding='SAME')
        return tf.nn.relu(tf.nn.bias_add(cov,b), name=name)

    def max_pool(self,name, l_input, k, s):
        return tf.nn.max_pool(l_input, ksize=[1, k, k, 1], strides=[1, s, s, 1], padding='SAME', name=name)

    def norm(self,name, l_input, lsize=4):
        return tf.nn.lrn(l_input, lsize, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name=name)

    def alex_net(self,_X, _dropout, n_classes, imagesize, img_channel):
        # Store layers weight & bias
        self._weights = {
            'wc1': tf.Variable(tf.random_normal([5, 5, img_channel, 64])),
            'wc2': tf.Variable(tf.random_normal([5, 5, 64, 64])),
            'wd1': tf.Variable(tf.random_normal([6*6*64, 256])),
            'wd2': tf.Variable(tf.random_normal([256, 128])),
            'wd3': tf.Variable(tf.random_normal([128, 10])),
            'out': tf.Variable(tf.random_normal([10, n_classes]))
        }
        
        self._biases = {
            'bc1': tf.Variable(tf.random_normal([64])),
            'bc2': tf.Variable(tf.random_normal([64])),
            'bd1': tf.Variable(tf.random_normal([256])),
            'bd2': tf.Variable(tf.random_normal([128])),
            'bd3': tf.Variable(tf.random_normal([10])),
            'out': tf.Variable(tf.random_normal([n_classes]))
        }
        # Reshape input picture
        _X = tf.reshape(_X, shape=[-1, imagesize, imagesize, img_channel])

        # Convolution Layer
        self.conv1 = self.conv1st('conv1', _X, self._weights['wc1'], self._biases['bc1'])

        # Max Pooling (down-sampling)
        self.pool1 = self.max_pool('pool1', self.conv1, k=2, s=2)
        # Apply Normalization
        self.norm1 = self.norm('norm1', self.pool1, lsize=4)
        # Apply Dropout
        self.norm1 = tf.nn.dropout(self.norm1, _dropout)

        # Convolution Layer
        self.conv2 = self.conv2d('conv2', self.norm1, self._weights['wc2'], self._biases['bc2'])
        # Max Pooling (down-sampling)
        self.pool2 = self.max_pool('pool2', self.conv2, k=2, s=2)
        # Apply Normalization
        self.norm2 = self.norm('norm2', self.pool2, lsize=4)
        # Apply Dropout
        self.norm2 = tf.nn.dropout(self.norm2, _dropout)

        '''# Convolution Layer
        self.conv3 = self.conv2d('conv3', self.norm2, self._weights['wc3'], self._biases['bc3'])
        self.conv4 = self.conv2d('conv4', self.conv3, self._weights['wc4'], self._biases['bc4'])
        self.conv5 = self.conv2d('conv5', self.conv4, self._weights['wc5'], self._biases['bc5'])
        # Max Pooling (down-sampling)
        self.pool3 = self.max_pool('pool3', self.conv5, k=3, s=2)
        # Apply Normalization
        self.norm3 = self.norm('norm3', self.pool3, lsize=4)
        # Apply Dropout
        self.norm3 = tf.nn.dropout(self.norm3, _dropout)'''
        
        # Fully connected layer
        layer_shape = self.norm2.get_shape()
        #print layer_shape
        num_features = layer_shape[1:4].num_elements()
        #print num_features
        self.dense1 = tf.reshape(self.norm2, [-1, num_features])
        #print self.dense1.get_shape()
        self.dense1 = tf.nn.relu(tf.matmul(self.dense1, self._weights['wd1']) + self._biases['bd1'], name='fc1') # Relu activation

        self.dense2 = tf.nn.relu(tf.matmul(self.dense1, self._weights['wd2']) + self._biases['bd2'], name='fc2') # Relu activation

        self.dense3 = tf.nn.relu(tf.matmul(self.dense2, self._weights['wd3']) + self._biases['bd3'], name='fc3') # Relu activation

        # Output, class prediction
        self.out = tf.matmul(self.dense3, self._weights['out']) + self._biases['out']
        return self.out
