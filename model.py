import time
import tensorflow as tf
import numpy as np
import os
from tensorflow.contrib import rnn
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

save_path = '/home/soya/Desktop'

def get_model(input, is_training):
    # input: bx10x227x227
    with tf.variable_scope('spatial_encoder_1'):
        input = tf.expand_dims(input, axis=-1)
        print(input)
        encoded = tf.layers.conv3d(input, 128, kernel_size=[1, 11, 11], strides=[1, 4, 4],
                                   padding='valid', name='conv1')
        encoded = tf.layers.batch_normalization(encoded, training=is_training) # bx10x55x55x128
        encoded = tf.nn.tanh(encoded, name='tanh_1')

    with tf.variable_scope('spatial_encoder_2'):
        encoded = tf.layers.conv3d(encoded, 64, kernel_size=[1, 5, 5], strides=[1, 2, 2],
                                   padding='valid', name='conv2')
        encoded = tf.layers.batch_normalization(encoded, training=is_training) # bx10x26x26x64
        encoded = tf.nn.tanh(encoded, name='tanh_2')

    with tf.variable_scope('temporal_encoder_1'):
        lstm_cell = rnn.ConvLSTMCell(2, input_shape=[26, 26, 64], output_channels=64, kernel_shape=[3, 3])
        output, _ = tf.nn.dynamic_rnn(lstm_cell, encoded, initial_state=None, dtype='float32')

    with tf.variable_scope('temporal_encoder_2'):
        lstm_cell = rnn.ConvLSTMCell(2, input_shape=[26, 26, 64], output_channels=32, kernel_shape=[2, 2])
        output, _ = tf.nn.dynamic_rnn(lstm_cell, output, initial_state=None, dtype='float32')
        #print (output)

    with tf.variable_scope('temporal_decoder_1'):
        lstm_cell = rnn.ConvLSTMCell(2, input_shape=[26, 26, 32], output_channels=64, kernel_shape=[3, 3])
        output, _ = tf.nn.dynamic_rnn(lstm_cell, output, initial_state=None, dtype='float32')
        print (output)

    with tf.variable_scope('spatial_decoder_1'):
        decoded = tf.layers.conv3d_transpose(output, 128, kernel_size=[1, 5, 5],
                                             padding='valid', strides=[1, 2, 2], name='deconv1')
        decoded = tf.layers.batch_normalization(decoded, training=is_training)
        decoded = tf.nn.tanh(decoded, name='tanh_3')
        #print(decoded)

    with tf.variable_scope('spatial_decoder_2'):
        decoded = tf.layers.conv3d_transpose(decoded, 1, kernel_size=[1, 11, 11],
                                             padding='valid', strides=[1, 4, 4], name='deconv2')
        #print(decoded)

    decoded = tf.squeeze(decoded, axis=-1)
    print (decoded)
    return decoded

def get_loss(decoded, gt):
    """decoded: Bx10x227x227
       gt:Bx10x227x227"""
    batch_size = gt.get_shape()[0].value
    pixel_costs = []
    max_score = min_score = 0
    # transform tensor to numpy array
    #sess = tf.Session()
    #origin = gt.eval(session=sess)
    #pred = decoded.eval(session=sess)
    #
    for volume in range(batch_size):
        #score = np.linalg.norm(decoded[volume, :] - gt[volume, :])
        score = tf.sqrt(tf.reduce_sum((decoded[volume, :] - gt[volume, :])**2)) # get L2 distance
        if volume == 0:
            max_score = min_score = score
        else:
            max_condition = tf.greater(score, max_score)  # which one is larger
            min_condition = tf.less(score, min_score)  # which one is smaller
            max_score = tf.where(max_condition, score, max_score)
            min_score = tf.where(min_condition, score, min_score)
        pixel_costs.append(score)
    print (pixel_costs)
    #final_cost = tf.reduce_mean(pixel_costs)
    #pixel_costs = tf.convert_to_tensor(pixel_costs, dtype=tf.float32) # return numpy to tensor
    return pixel_costs, max_score, min_score

def get_loss_MSE(decoded, gt):
    """decoded: Bx10x227x227
           gt:Bx10x227x227"""
    mse = tf.reduce_mean(tf.square(decoded - gt))
    print(mse)
    return mse

def get_loss_L2(decoded, gt):
    L2_loss = tf.sqrt(tf.reduce_sum(tf.square((decoded - gt))))
    print(L2_loss)
    return L2_loss

if __name__ == '__main__':
    with tf.Graph().as_default():
        test = tf.placeholder(tf.float32, shape=(64, 10, 227, 227))
        is_training_pl = tf.placeholder(dtype=tf.bool, shape=())
        pred = get_model(test, is_training_pl)
        #loss, min_score, max_score = get_loss(pred, test)
        loss = get_loss_L2(pred, test)
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            is_training = True
            sess.run(init)
            start = time.time()
            for i in range(1):
                print (i)
                #sess.run(pred, feed_dict={test: np.random.rand(64, 10, 227, 227), is_training_pl: is_training})
                pred, loss = sess.run([pred, loss], feed_dict={test: np.random.rand(64, 10, 227, 227),
                                                               is_training_pl: is_training})
                #np.save(os.path.join(save_path, 'pred.npy'), pred)
                #print(loss)
                print(pred.shape)
            print(time.time()-start)
