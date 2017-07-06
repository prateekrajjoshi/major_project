#! /usr/bin/env python

# TYO PARAMETERS HARU ADJUST GARNA MATRA BAAKI HO...... like image size in this script and on model_alexNet.py 

import pickle_load
import model_alexNet
import tensorflow as tf
import img_proc
import os
import time

data = pickle_load.Dataset('/home/aashish/Documents/cifar-10-batches-py')

# Parameters
learn_rate = 0.001
decay_rate = 0.1
batch_size = 128
display_step = 20

#n_classes = data.total_data # we got mad kanji
n_classes = 10
dropout = 0.8 # Dropout, probability to keep units
imagesize = 24
img_channel = 3
inference = model_alexNet.modelAlexNet()
x = tf.placeholder(tf.float32, [None, imagesize, imagesize, img_channel])
#distorted_images = img_proc.pre_process(images=x, training=True)
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32) # dropout (keep probability)

pred = inference.model_predict(x, keep_prob, n_classes, imagesize, img_channel)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))

global_step = tf.Variable(initial_value=0,name = 'global_step', trainable=False)
lr = tf.train.exponential_decay(learn_rate, global_step, 1000, decay_rate, staircase=True)
#lr =  1e-2
optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost, global_step=global_step)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

saver = tf.train.Saver()
tf.add_to_collection("x", x)
tf.add_to_collection("y", y)
tf.add_to_collection("keep_prob", keep_prob)
tf.add_to_collection("pred", pred)
tf.add_to_collection("accuracy", accuracy)

save_dir = 'save/'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)


init = tf.global_variables_initializer()
num_iterations = 1000

with tf.Session() as sess:
	try:
		print("Trying to restore last checkpoint ...")
		# Use TensorFlow to find the latest checkpoint - if any.
		last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=save_dir)

		# Try and load the data in the checkpoint.
		saver.restore(sess, save_path=last_chk_path)

		# If we get to this point, the checkpoint was successfully loaded.
		print("Restored checkpoint from:", last_chk_path)
	except:
		# If the above failed for some reason, simply
		# initialize all the variables for the TensorFlow graph.
		print("Failed to restore checkpoint. Initializing variables instead.")
		sess.run(init)

	for i in range(num_iterations):
		start_time = time.time()
		batch_xs, batch_ys, batch_yhot = data.getNextBatch(batch_size)
		batch_xs = img_proc.pre_process(batch_xs).eval()
		#i_global, _ = sess.run([global_step, optimizer], feed_dict={x: batch_xs, y: batch_yhot, keep_prob: dropout})

		sess.run(optimizer, feed_dict={x: batch_xs, y: batch_yhot})#, keep_prob: dropout})
		i_global = tf.train.global_step(sess, global_step)
		print i_global

		if (i_global % display_step == 0) or (i == num_iterations - 1):
			acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_yhot})#, keep_prob: dropout})
			loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_yhot})#, keep_prob: dropout})
			#rate = sess.run(lr)
			rate = lr

			print "lr " + str(rate) + " Iter " + str(i) + ", Minibatch Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc)
		
		if (i_global % (5 * display_step) == 0) or (i == num_iterations - 1):
			saver.save(sess, os.path.join(save_dir, 'cifar10_nn'), global_step=global_step)
			print("Saved checkpoint.")

    	# Ending time.
    	end_time = time.time()

    	# Difference between start and end-times.
    	time_dif = end_time - start_time

    	# Print the time-usage.
	   	#print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))
	   	#print ("Time used: %d" + (timedelta(seconds=int(round(time_dif)))))
	#print "Optimization Finished!"
	'''step_test = 1
	while step_test * batch_size < len(testing):
	testing_ys, testing_xs = testing.nextBatch(batch_size)
	print "Testing Accuracy:", sess.run(accuracy, feed_dict={x: testing_xs, y: testing_ys, keep_prob: 1.})
	step_test += 1 '''

