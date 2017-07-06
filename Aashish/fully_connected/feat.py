import os
import time
import cPickle
import classify_image
import numpy as np
import tensorflow as tf

def unpickle(file):
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

def extract_features(file_name):
    step = 1
    d = unpickle(os.path.join(os.path.expanduser('cifar-10-batches-py/'), file_name))
    data = d['data']
    imgs = np.transpose(np.reshape(data,(-1,32,32,3), order='F'),axes=(0,2,1,3)) #order batch,x,y,color
    y = np.asarray(d['labels'], dtype='uint8')
    yn = y[0: int(len(y)/step)]
    

    '''#Debug codes
    y_limited = np.zeros((10,))
    for x in y[range(0,len(y),1000)]:
        y_limited[x] = y[x]'''

    FLAGS = classify_image.parseArg()
    classify_image.maybe_download_and_extract(FLAGS.model_dir)
    classify_image.create_graph(FLAGS.model_dir)
    with tf.Session() as sess:
        #softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')
        representation_tensor = sess.graph.get_tensor_by_name('pool_3:0')
        #predictions = np.zeros((len(y), 1008), dtype='float32')
        #representations = np.zeros((len(y), 2048), dtype='float32')
        
        #Debug Codes
        transferVal = np.zeros((int(len(y)/step), 2048), dtype='float32')

        for i in range(0,len(yn)):
            start = time.time()
            #[reps, preds] = sess.run([representation_tensor, softmax_tensor], {'DecodeJpeg:0': imgs[i]})
            reps = sess.run(representation_tensor, {'DecodeJpeg:0' : imgs[i]})
            #if (i % 10 == 0):
            if True:
                print("{}/{} Time for image {} ".format(i+1, len(yn), time.time() - start))
            #squeze to remove the dimention with shape value of 1 ( (1,1,3) chha bhane s0queeze pacchi (3,) hun6 shape; 2nd argument can be axis = (n,) kun chai shape position lai hataune ho )
            #predictions[i] = np.squeeze(preds)
            #representations[i] = np.squeeze(reps)

            #Debug Code
            transferVal[i] = np.squeeze(reps)
        #np.savez_compressed(file_name + ".npz", predictions=predictions, representations=representations, y=y)
        np.savez_compressed(file_name + ".npz", representations=transferVal, y=yn)

if __name__ == '__main__':
    #extract_features('test_batch')
    #extract_features('data_batch_1')
    #extract_features('data_batch_2')
    extract_features('data_batch_3')
    #extract_features('data_batch_4')
    #extract_features('data_batch_5')




