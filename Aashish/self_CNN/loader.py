import tensorflow as tf 
import os
import numpy as np

#~==DEBUG_CODES==
from PIL import Image

IMAGE_SIZE = 32
IMAGE_CHANNELS = 1
image_path = "new6.jpg"
image_data = tf.gfile.FastGFile(image_path, 'rb').read()
#print image_data
with tf.Session().as_default():
	rgb_image = tf.image.decode_jpeg(image_data, channels=IMAGE_CHANNELS)
	rgb_image = tf.image.resize_images(rgb_image, [IMAGE_SIZE, IMAGE_SIZE])
	rgb_image = tf.reshape(rgb_image, [IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS]).eval()
pil_image = Image.fromarray(rgb_image)
print rgb_image.shape
#pil_image.show()
rgb_image = np.array(np.expand_dims(rgb_image , axis = 0))

print rgb_image.dtype
#print type(rgb_image)
#rgb_image = rgb_image.astype(float32)
with tf.Session() as sess:
	new_saver = tf.train.import_meta_graph('model.meta')
	new_saver.restore(sess, tf.train.latest_checkpoint('./'))
	# Feed the image_data as input to the graph and get first prediction
	softmax_tensor = sess.graph.get_tensor_by_name('output_layer:0')
	print softmax_tensor
	predictions = sess.run(softmax_tensor,feed_dict={'x_input:0':rgb_image,'keep_prob:0':1})
	print predictions
