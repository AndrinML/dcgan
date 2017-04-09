"""
Andrin Jenal, 2017
ETH Zurich
"""

from tensorflow.contrib import layers
import tensorflow as tf


def leaky_relu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)


def leaky_relu2(x, alpha=0.2):
    return tf.maximum(tf.minimum(0.0, alpha * x), x)


def leaky_relu_batch_norm(x, alpha=0.2):
    return leaky_relu(layers.batch_norm(x, decay=0.9), alpha)


def relu_batch_norm(x):
    return tf.nn.relu(layers.batch_norm(x, decay=0.9))


def linear(x, output_size, scope="linear", stddev=0.02):
    with tf.variable_scope(scope):
        weights = tf.get_variable("weights", [x.get_shape()[1], output_size], dtype=tf.float32, initializer=tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size], initializer=tf.constant_initializer(0.0))
        return tf.matmul(x, weights) + bias


def linear_contrib(inputs, output_size, activation_fn=None, scope="linear", stddev=0.02):
    with tf.variable_scope(scope):
        return layers.fully_connected(inputs, output_size,
                                      activation_fn=activation_fn,
                                      weights_initializer=tf.random_normal_initializer(stddev=stddev),
                                      biases_initializer=tf.constant_initializer(0.0))


def binary_crossentropy(x, x_recons, eps=1e-8):
    return -(x * tf.log(x_recons + eps) + (1.0 - x) * tf.log(1.0 - x_recons + eps))


def binary_crossentropy_with_clip(x, x_hat, eps=1e-8):
    return -(x * tf.log(tf.clip_by_value(x_hat, eps, 1.0)) + (1.0 - x) * tf.log(tf.clip_by_value(1.0 - x_hat, eps, 1.0)))


def conv2d_contrib(inputs, output_dim, kernel=3, stride=1, padding="SAME", activation_fn=tf.nn.relu, scope="conv2d"):
    with tf.variable_scope(scope):
        return layers.conv2d(inputs, output_dim, kernel, stride=stride, padding=padding, activation_fn=activation_fn)


def conv2d_transpose_contrib(inputs, output_dim, kernel=3, stride=1, padding="SAME", activation_fn=tf.nn.relu, scope="conv2d_transpose"):
    with tf.variable_scope(scope):
        return layers.conv2d_transpose(inputs, output_dim, kernel, stride=stride, padding=padding, activation_fn=activation_fn)


def avg_pool_contrib(inputs, scope="pool2d"):
    return layers.avg_pool2d(inputs, kernel_size=2, scope=scope)


def flatten_contrib(inputs):
    return layers.flatten(inputs)


def clip_gradient_norms(gradients, clip_norm):
    for i, (g, v) in enumerate(gradients):
        if g is not None:
            gradients[i] = (tf.clip_by_norm(g, clip_norm), v)
    return gradients


def gradient_summary(grad_norms):
    with tf.name_scope("gradients"):
        tf.summary.histogram("gradient_l2_norms", grad_norms)
        tf.summary.scalar("gradient_l2_norms_mean", tf.reduce_mean(tf.convert_to_tensor(grad_norms)))
