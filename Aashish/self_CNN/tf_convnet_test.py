import tensorflow as tf
import numpy as np
import os
import time
import math
from tensorflow.python.framework import graph_util

#~==DEBUG_CODES==~
from PIL import Image


# hyper parameters to use for training
TRAIN_BATCH_SIZE = 10
TRAIN_EPOCHS = 5
VALID_BATCH_SIZE = 40
VALID_EPOCHS = None
SHUFFLE_BATCHES = True
LEARNING_RATE = 0.01
for line in open('num_class.txt','r'):
	NUM_CLASSES = int(line)
KEEP_PROB = 0.75
VALID_STEPS = 10

# image parameters
IMAGE_SIZE = 32
IMAGE_CHANNELS = 1

def get_image_label_list(image_label_file):
    filenames = []
    labels = []
    for line in open(image_label_file, "r"):
        filename, label = line[:-1].split(' ')
        filenames.append(filename)
        labels.append(int(label))

    # == debug ==
    print "get_image_label_list: read " + str(len(filenames)) \
        + " items"
    # == /debug ==

    return filenames, labels

def read_image_from_disk(input_queue):
    label = input_queue[1]
    print label
    file_contents = tf.read_file(input_queue[0])
    rgb_image = tf.image.decode_jpeg(file_contents, channels=IMAGE_CHANNELS,
        name="decode_jpeg")
    rgb_image = tf.image.resize_images(rgb_image, [IMAGE_SIZE, IMAGE_SIZE]) 

    '''### ~==DEBUG_CODES==~
    sess = tf.Session()
    with sess.as_default():
        rgb = rgb_image[0].eval()
        rgb.show()'''

    return rgb_image, label

def inputs(train_file, batch_size=TRAIN_BATCH_SIZE, num_epochs=TRAIN_EPOCHS):
    image_list, label_list = get_image_label_list(train_file)
    input_queue = tf.train.slice_input_producer([image_list, label_list],
        num_epochs=num_epochs, shuffle=SHUFFLE_BATCHES)
    image, label = read_image_from_disk(input_queue)
    image = tf.reshape(image, [IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS])
    image_batch, label_batch = tf.train.batch([image, label],
        batch_size=batch_size)

    return image_batch, tf.one_hot(tf.to_int64(label_batch),
        NUM_CLASSES, on_value=1.0, off_value=0.0)

def conv2d(x, W, b, strides=1,name=''):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1],
        padding='SAME',name=name)
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def maxpool2d(x, k=2, layer=""):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1],
        strides=[1, k, k, 1], padding='SAME')

def conv_net(x, weights, biases, image_size, keep_prob=KEEP_PROB):
    # Convolution and max pooling layers
    # Each max pooling layer reduces dimensionality by 2

#with tf.name_scope('layer1'):
    # Convolution and max pooling layer 1
    conv1 = conv2d(x, weights['wc1'], biases['bc1'],1,'conv1')
    conv1 = maxpool2d(conv1, k=2)

#with tf.name_scope('layer2'):
    # Convolution and max pooling layer 2
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'],1,'conv2')
    conv2 = maxpool2d(conv2, k=2)

#with tf.name_scope('layer3'):
    # Convolution and max pooling layer 3
    conv3 = conv2d(conv2, weights['wc3'], biases['bc3'],1,'conv3')
    conv3 = maxpool2d(conv3, k=2)

#with tf.name_scope('layer4'):
    # Convolution and max pooling layer 4
    conv4 = conv2d(conv3, weights['wc4'], biases['bc4'],1,'conv4')
    conv4 = maxpool2d(conv4, k=2)

#with tf.name_scope('fully_connected'):
    # Fully-connected layer
    fc1 = tf.reshape(conv4, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)

    # Apply dropout
    fc1 = tf.nn.dropout(fc1, keep_prob)

#with tf.name_scope('output'):
    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])

    final_tensor = tf.nn.softmax(out, name='output_layer')
    return out, final_tensor

def generate_image_summary(x, weights, biases, step, image_size=IMAGE_SIZE):
    with tf.name_scope('generate_image_summary'):
        x =  tf.slice(x, [0, 0, 0, 0],
            [VALID_BATCH_SIZE, image_size, image_size, IMAGE_CHANNELS])
        x = tf.nn.conv2d(x, weights['wc1'], strides=[1, 1, 1, 1],
            padding='SAME')

        # Nifty grid image summary via:
        # http://stackoverflow.com/questions/33802336/visualizing-output-of-convolutional-layer-in-tensorflow

        x = tf.slice(x, [0, 0, 0, 0], [1, -1, -1, -1])
        x = tf.reshape(x, [IMAGE_SIZE, IMAGE_SIZE, 32])

        pad_xy = image_size + 4
        x = tf.image.resize_image_with_crop_or_pad(x, pad_xy, pad_xy)
        x = tf.reshape(x, [pad_xy, pad_xy, 4, 8])
        x = tf.transpose(x, [2, 0, 3, 1])
        x = tf.reshape(x, [1, pad_xy * 4, pad_xy * 8, 1])

        conv_summary = tf.summary.image("img_conv_{:05d}".format(step), x)
        relu_summary = tf.summary.image("img_relu_{:05d}".format(step), tf.nn.relu(x))

    return conv_summary, relu_summary

def main(argv=None):

    # Read inventory of training images and labels
    with tf.name_scope('batch_inputs'):
        train_file = "./train.txt"
        valid_file = "./valid.txt"

        image_size = IMAGE_SIZE

        train_image_batch, train_label_batch = inputs(train_file,
            batch_size=TRAIN_BATCH_SIZE, num_epochs=TRAIN_EPOCHS)
        valid_image_batch, valid_label_batch = inputs(valid_file,
            batch_size=VALID_BATCH_SIZE, num_epochs=VALID_EPOCHS)

    # These are image and label batch placeholders which we'll feed in during training
    x_ = tf.placeholder("float32", shape=[None, image_size, image_size,
        IMAGE_CHANNELS],name='x_input')

    y_ = tf.placeholder("float32", shape=[None, NUM_CLASSES])

    # k is the image size after 4 convolution layers
    k = int(math.ceil(IMAGE_SIZE / 2.0 / 2.0 / 2.0 / 2.0))

    # Store weights for our convolution & fully-connected layers
    with tf.name_scope('weights'):
        weights = {
            # 5x5 conv, 3 input channel, 32 outputs each
            'wc1': tf.Variable(tf.random_normal([5, 5, 1 * IMAGE_CHANNELS, 32])),
            # 5x5 conv, 32 inputs, 64 outputs
            'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
            # 5x5 conv, 64 inputs, 128 outputs
            'wc3': tf.Variable(tf.random_normal([5, 5, 64, 128])),
            # 5x5 conv, 128 inputs, 256 outputs
            'wc4': tf.Variable(tf.random_normal([5, 5, 128, 256])),
            # fully connected, k * k * 256 inputs, 1024 outputs
            'wd1': tf.Variable(tf.random_normal([k * k * 256, 1024])),
            # 1024 inputs, 2 class labels (prediction)
            'out': tf.Variable(tf.random_normal([1024, NUM_CLASSES]))
        }

    # Store biases for our convolution and fully-connected layers
    with tf.name_scope('biases'):
        biases = {
            'bc1': tf.Variable(tf.random_normal([32])),
            'bc2': tf.Variable(tf.random_normal([64])),
            'bc3': tf.Variable(tf.random_normal([128])),
            'bc4': tf.Variable(tf.random_normal([256])),
            'bd1': tf.Variable(tf.random_normal([1024])),
            'out': tf.Variable(tf.random_normal([NUM_CLASSES]))
        }

    # Define dropout rate to prevent overfitting
    keep_prob = tf.placeholder(tf.float32,name='keep_prob')

    # Build our graph
    pred, final_layer = conv_net(x_, weights, biases, image_size, keep_prob)
    # Calculate loss
    with tf.name_scope('cross_entropy'):
        # Define loss and optimizer
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y_))
        cost_summary = tf.summary.scalar("cost_summary", cost)

    # Run optimizer step
    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cost)

    # Evaluate model accuracy
    with tf.name_scope('predict'):
        correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        accuracy_summary = tf.summary.scalar("accuracy_summary", accuracy)
        w_summary = tf.summary.histogram("weights", weights['wc1'])
        b_summary = tf.summary.histogram("biases", biases['bc1'])

    sess = tf.Session()

    writer = tf.summary.FileWriter("./logs", sess.graph)

    init_op = tf.global_variables_initializer()
    init_local_op = tf.local_variables_initializer()

    saver = tf.train.Saver(max_to_keep=1)

    step = 0

    with sess.as_default():
        sess.run(init_op)
        sess.run(init_local_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            while not coord.should_stop():
                step += 1
                x, y = sess.run([train_image_batch, train_label_batch])
                train_step.run(feed_dict={keep_prob: 0.75,
                    x_: x, y_: y})

                #~==DEBUG_CODES==~
                #x = x_.eval()
                #x[0].show()



                if step % VALID_STEPS == 0:
                    x, y = sess.run([valid_image_batch, valid_label_batch])
                    conv_summary, relu_summary = generate_image_summary(x_, weights, biases, step, image_size)
                    result = sess.run([cost_summary, accuracy_summary, accuracy, conv_summary, relu_summary, w_summary, b_summary],
                        feed_dict={keep_prob: 1.0, x_: x, y_: y})

                    cost_summary_str = result[0]
                    accuracy_summary_str = result[1]
                    acc = result[2]
                    conv_summary_str = result[3]
                    relu_summary_str = result[4]
                    w_summary_str = result[5]
                    b_summary_str = result[6]

                    # write summaries for viewing in Tensorboard
                    writer.add_summary(accuracy_summary_str, step)
                    writer.add_summary(cost_summary_str, step)
                    writer.add_summary(conv_summary_str, step)
                    writer.add_summary(relu_summary_str, step)
                    writer.add_summary(w_summary_str, step)
                    writer.add_summary(b_summary_str, step)

                    print("Accuracy at step %s: %s" % (step, acc))

                  

        except tf.errors.OutOfRangeError:
            x, y = sess.run([valid_image_batch, valid_label_batch])
            result = sess.run([accuracy], feed_dict={keep_prob: 1.0,
                x_: x, y_: y})
            print("Validation accuracy: %s" % result[0])

        finally:
            #os.chdir(path)
            coord.request_stop()
            coord.join(threads)
            save_path = saver.save(sess, "./model")

            sess.close()

    return 0

if __name__ == '__main__':
    tf.app.run()
