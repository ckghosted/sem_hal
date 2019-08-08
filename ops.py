import math
import numpy as np 
import tensorflow as tf

from tensorflow.python.framework import ops

from utils import *

try:
    image_summary = tf.image_summary
    scalar_summary = tf.scalar_summary
    histogram_summary = tf.histogram_summary
    merge_summary = tf.merge_summary
    SummaryWriter = tf.train.SummaryWriter
except:
    image_summary = tf.summary.image
    scalar_summary = tf.summary.scalar
    histogram_summary = tf.summary.histogram
    merge_summary = tf.summary.merge
    SummaryWriter = tf.summary.FileWriter

if "concat_v2" in dir(tf):
    def concat(tensors, axis, *args, **kwargs):
        return tf.concat_v2(tensors, axis, *args, **kwargs)
else:
    def concat(tensors, axis, *args, **kwargs):
        return tf.concat(tensors, axis, *args, **kwargs)

class batch_norm(object):
    def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon  = epsilon
            self.momentum = momentum
            self.name = name
    def __call__(self, x, train=True, reuse=False):
        return tf.contrib.layers.batch_norm(x,
                                            decay=self.momentum,
                                            updates_collections=None,
                                            epsilon=self.epsilon,
                                            scale=True,
                                            is_training=train,
                                            scope=self.name,
                                            reuse=reuse)

def conv_cond_concat(x, y):
    """Concatenate conditioning vector on feature map axis."""
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()
    return concat([x, y*tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], 3)

def conv2d(input_, output_dim, 
           k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
           add_bias=False,
           name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')
        if add_bias:
            biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
            # conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
            conv = tf.nn.bias_add(conv, biases)
        
        return conv

def deconv2d(input_, output_dim,
             k_h=4, k_w=4, d_h=2, d_w=2, stddev=0.02,
             add_bias=False,
             name="deconv2d", with_w=False):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_w, output_dim[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))
        try:
            deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_dim, strides=[1, d_h, d_w, 1])
        # Support for verisons of TensorFlow before 0.7.0
        except AttributeError:
            deconv = tf.nn.deconv2d(input_, w, output_shape=output_dim, strides=[1, d_h, d_w, 1])
        
        if add_bias:
            biases = tf.get_variable('biases', [output_dim[-1]], initializer=tf.constant_initializer(0.0))
            # deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())
            deconv = tf.nn.bias_add(deconv, biases)
        
        if with_w and add_bias:
            return deconv, w, biases
        elif with_w:
            return deconv, w
        else:
            return deconv

def lrelu(x, leak=0.01, name="lrelu"):
    return tf.maximum(x, leak*x)

def linear(input_, output_size, stddev=0.02, bias_start=0.0,
           add_bias=False,
           name="linear", with_w=False):
    shape = input_.get_shape().as_list()
    with tf.variable_scope(name):
        if len(shape) == 2:
            matrix = tf.get_variable("Matrix", [shape[-1], output_size], tf.float32,
                                     tf.random_normal_initializer(stddev=stddev))
            if add_bias:
                bias = tf.get_variable("bias", [output_size],
                                       initializer=tf.constant_initializer(bias_start))
            if add_bias:
                if with_w:
                    return tf.matmul(input_, matrix) + bias, matrix, bias
                else:
                    return tf.matmul(input_, matrix) + bias
            else:
                if with_w:
                    return tf.matmul(input_, matrix), matrix
                else:
                    return tf.matmul(input_, matrix)
        elif len(shape) == 3:
            matrix = tf.get_variable("Matrix", [output_size, shape[-1], 1], tf.float32,
                                     tf.random_normal_initializer(stddev=stddev))
            if add_bias:
                bias = tf.get_variable("bias", [output_size, 1],
                                       initializer=tf.constant_initializer(bias_start))
            if add_bias:
                if with_w:
                    return tf.matmul(input_, matrix) + tf.reshape(tf.tile(bias, [shape[1], 1]), [-1, shape[1], 1]), matrix, bias
                else:
                    return tf.matmul(input_, matrix) + tf.reshape(tf.tile(bias, [shape[1], 1]), [-1, shape[1], 1])
            else:
                if with_w:
                    return tf.matmul(input_, matrix), matrix
                else:
                    return tf.matmul(input_, matrix)

## linear layer with the matrix being initialized by identity matrix (maybe some zero rows or columns if not square matrix)
def linear_identity(input_, output_size, stddev=0.02, bias_start=0.0,
           add_bias=False,
           name="linear", with_w=False):
    shape = input_.get_shape().as_list()
    with tf.variable_scope(name):
        if len(shape) == 2:
            if shape[-1] >= output_size:
                init_matrix = np.concatenate((np.identity(output_size), np.zeros((shape[-1] - output_size, output_size))), axis=0)
            else:
                init_matrix = np.identity(output_size)[0:shape[-1], :]
            matrix = tf.get_variable("Matrix", [shape[-1], output_size], tf.float32,
                                     initializer=tf.constant_initializer(init_matrix))
            if add_bias:
                bias = tf.get_variable("bias", [output_size],
                                       initializer=tf.constant_initializer(bias_start))
            if add_bias:
                if with_w:
                    return tf.matmul(input_, matrix) + bias, matrix, bias
                else:
                    return tf.matmul(input_, matrix) + bias
            else:
                if with_w:
                    return tf.matmul(input_, matrix), matrix
                else:
                    return tf.matmul(input_, matrix)
        elif len(shape) == 3:
            matrix = tf.get_variable("Matrix", [output_size, shape[-1], 1], tf.float32,
                                     tf.random_normal_initializer(stddev=stddev))
            if add_bias:
                bias = tf.get_variable("bias", [output_size, 1],
                                       initializer=tf.constant_initializer(bias_start))
            if add_bias:
                if with_w:
                    return tf.matmul(input_, matrix) + tf.reshape(tf.tile(bias, [shape[1], 1]), [-1, shape[1], 1]), matrix, bias
                else:
                    return tf.matmul(input_, matrix) + tf.reshape(tf.tile(bias, [shape[1], 1]), [-1, shape[1], 1])
            else:
                if with_w:
                    return tf.matmul(input_, matrix), matrix
                else:
                    return tf.matmul(input_, matrix)