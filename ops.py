import tensorflow as tf
import tensorflow.contrib.layers as layers
import tensorflow.contrib.slim as slim
import numpy as np

def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

def selu(x):
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return scale*tf.where(x>0.0, x, alpha*tf.exp(x)-alpha)

def huber_loss(labels, predictions, delta=1.0):
    residual = tf.abs(predictions - labels)
    condition = tf.less(residual, delta)
    small_res = 0.5 * tf.square(residual)
    large_res = delta * residual - 0.5 * tf.square(delta)
    return tf.where(condition, small_res, large_res)

def conv2d(input, output_shape, is_train, act,  k_h=5, k_w=5, stddev=0.02, name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input.get_shape()[-1], output_shape],
                initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input, w, strides=[1, 2, 2, 1], padding='SAME')

        biases = tf.get_variable('biases', [output_shape], initializer=tf.constant_initializer(0.0))
        if act == 'relu':
            conv = tf.nn.relu(tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape()))
        elif act == 'lrelu':
            conv = lrelu(tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape()))
        elif act == 'selu':
            conv = selu(tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape()))
        else:
            raise ValueError('Please choose a valid activation fn.')
        bn = tf.contrib.layers.batch_norm(conv, center=True, scale=True, 
            decay=0.9, is_training=is_train, updates_collections=None)
    return bn

def fc(input, output_shape, is_train, act, name="fc"):
    if act == 'relu':
        output = slim.fully_connected(input, output_shape, activation_fn=tf.nn.relu)
    elif act == 'lrelu':
        output = slim.fully_connected(input, output_shape, activation_fn=lrelu)
    elif act == 'selu':
        output = slim.fully_connected(input, output_shape, activation_fn=selu)
    else:
        raise ValueError('Please choose a valid activation fn.')
    return output
