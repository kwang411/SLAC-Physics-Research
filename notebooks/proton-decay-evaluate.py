from __future__ import division
# Preparation: make symbolic links for practice_train_10k.root and practice_test_10k.root
import subprocess
# ln -sf ../../mix.root ./train_pdecay.root
# ln -sf ../../practice_test_5k.root ./test_pdecay.root

from larcv import larcv
from larcv.dataloader2 import larcv_threadio
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os,sys,time

# tensorflow/gpu start-up configuration
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import tensorflow as tf

TUTORIAL_DIR     = '..'
TRAIN_IO_CONFIG  = os.path.join(TUTORIAL_DIR, 'tf/pdecay_train.cfg')
TEST_IO_CONFIG   = os.path.join(TUTORIAL_DIR, 'tf/pdecay_test.cfg' )
TRAIN_BATCH_SIZE = 64
TEST_BATCH_SIZE  = 1024
LOGDIR           = 'log_pdecay'
ITERATIONS       = 5000
SAVE_SUMMARY     = 20
SAVE_WEIGHTS     = 100

#
# Step 0: IO
#
# for "train" data set
train_io = larcv_threadio()  # create io interface
train_io_cfg = {'filler_name' : 'TrainIO',
                'verbosity'   : 0,
                'filler_cfg'  : TRAIN_IO_CONFIG}
train_io.configure(train_io_cfg)   # configure
train_io.start_manager(TRAIN_BATCH_SIZE) # start read thread
time.sleep(2)
train_io.next()

# for "test" data set
test_io = larcv_threadio()   # create io interface
test_io_cfg = {'filler_name' : 'TestIO',
               'verbosity'   : 0,
               'filler_cfg'  : TEST_IO_CONFIG}
test_io.configure(test_io_cfg)   # configure
test_io.start_manager(TEST_BATCH_SIZE) # start read thread
time.sleep(2)
test_io.next()

#
# Step 1: Define network
#
import tensorflow.contrib.slim as slim
import tensorflow.python.platform

def build(input_tensor, num_class=4, trainable=True, debug=True):

    net = input_tensor
    if debug: print('input tensor:', input_tensor.shape)

    filters = 32
    num_modules = 5
    with tf.variable_scope('conv'):
        for step in xrange(5):
            stride = 2
            if step: stride = 1
            net = slim.conv2d(inputs        = net,        # input tensor
                              num_outputs   = filters,    # number of filters (neurons) = # of output feature maps
                              kernel_size   = [3,3],      # kernel size
                              stride        = stride,     # stride size
                              trainable     = trainable,  # train or inference
                              activation_fn = tf.nn.relu, # relu
                              normalizer_fn = slim.batch_norm,
			      scope         = 'conv%da_conv' % step)

            net = slim.conv2d(inputs        = net,        # input tensor
                              num_outputs   = filters,    # number of filters (neurons) = # of output feature maps
                              kernel_size   = [3,3],      # kernel size
                              stride        = 1,          # stride size
                              trainable     = trainable,  # train or inference
                              activation_fn = tf.nn.relu, # relu
			      normalizer_fn = slim.batch_norm,
                              scope         = 'conv%db_conv' % step)
            if (step+1) < num_modules:
                net = slim.max_pool2d(inputs      = net,    # input tensor
                                      kernel_size = [2,2],  # kernel size
                                      stride      = 2,      # stride size
                                      scope       = 'conv%d_pool' % step)

            else:
                net = tf.layers.average_pooling2d(inputs = net,
                                                  pool_size = [net.get_shape()[-2].value,net.get_shape()[-3].value],
                                                  strides = 1,
                                                  padding = 'valid',
                                                  name = 'conv%d_pool' % step)
            filters *= 2

            if debug: print('After step',step,'shape',net.shape)

    with tf.variable_scope('final'):
        net = slim.flatten(net, scope='flatten')

        if debug: print('After flattening', net.shape)

        net = slim.fully_connected(net, int(num_class), scope='final_fc')

        if debug: print('After final_fc', net.shape)

    return net

#
# Step 2: Build network + define loss & solver
#
# retrieve dimensions of data for network construction
dim_data  = train_io.fetch_data('train_image').dim()
dim_label = train_io.fetch_data('train_label').dim()
# define place holders
data_tensor    = tf.placeholder(tf.float32, [None, dim_data[1] * dim_data[2] * dim_data[3]], name='image')
label_tensor   = tf.placeholder(tf.float32, [None, dim_label[1]], name='label')
data_tensor_2d = tf.reshape(data_tensor, [-1,dim_data[1],dim_data[2],dim_data[3]],name='image_reshape')


# build net
net = build(input_tensor=data_tensor_2d, num_class=dim_label[1], trainable=True, debug=False)
# Define accuracy
with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(net,1), tf.argmax(label_tensor,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)
# Define loss + backprop as training step
with tf.name_scope('train'):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label_tensor, logits=net))
    tf.summary.scalar('cross_entropy',cross_entropy)
    train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)
                                                                                                                              

sess = tf.InteractiveSession()                                                                                                 
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(sess, "weights_pdecay/toynet-24999")

initial_loss = 0
initial_accuracy = 0
iterations = 20
for i in range(iterations):

    test_data  = test_io.fetch_data('test_image').data()
    test_label = test_io.fetch_data('test_label').data()
    feed_dict  = { data_tensor  : test_data,
                   label_tensor : test_label }
    loss, acc = sess.run([cross_entropy, accuracy], feed_dict=feed_dict)
    initial_loss += loss
    initial_accuracy += acc
    print("loss = " + str(loss) + ", accuracy = " + str(100*acc) + "%")
    train_io.next()
    test_io.next()

print("average loss = " + str(initial_loss/iterations) + ", average accuracy = " + str(100*initial_accuracy/iterations) + "%")
       
train_io.reset()
test_io.reset()
