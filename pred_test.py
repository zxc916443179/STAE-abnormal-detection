import tensorflow as tf
import os
import sys
import time
import matplotlib.pyplot as plt
import numpy as np

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)
import model, processing

h5_file_path = '/media/soya/Newsmy/datasets/UCSDped/h5_data/test_2/UCSDped1_testing_frames_Test001.h5'
MODEL_PATH = '/home/soya/Desktop/log_4'



with tf.Graph().as_default():
    test = tf.placeholder(tf.float32, shape=(1, 10, 227, 227))
    pred = model.get_model(test, False)

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        file = processing.loadDataFile(h5_file_path, 'eval', 0.5)
        current_data = file[60:61, :]
        sess.run(init)

        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(MODEL_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        print('model restored from %s' % (MODEL_PATH))

        pred = sess.run(pred, feed_dict={test: current_data})
        np.save('/home/soya/Desktop/pred_test_abnormal.npy', pred)
        start = time.time()
        print(time.time()-start)



