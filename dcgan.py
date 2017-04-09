"""
Andrin Jenal, 2017
ETH Zurich
"""


import tensorflow as tf
import numpy as np

import nn_ops


class DCGAN:

    def __init__(self, image_size, channels, z_size=256, learning_rate=5e-4):
        # summaries
        self.merged_summary_op = None
        self.summary_writer = None

        self.image_size = image_size
        self.image_channels = channels
        self.z_size = z_size
        self.learning_rate = learning_rate
        self.eps = 1e-8

        self.x = tf.placeholder(tf.float32, shape=(None, self.image_size, self.image_size, self.image_channels))
        self.batch_size = tf.shape(self.x)[0]
        self.z = tf.random_normal((self.batch_size, self.z_size), mean=0, stddev=1)

        with tf.variable_scope("dcgan_model"):

            with tf.variable_scope("generator"):
                self.generated_samples = self._generator(self.z)

            with tf.variable_scope("discriminator"):
                self.dis_x = self._discriminator(self.x)
                tf.summary.histogram("predicted_x_values", self.dis_x)

            with tf.variable_scope("discriminator", reuse=True):
                self.dis_x_pred = self._discriminator(self.generated_samples)
                tf.summary.histogram("predicted_x_z_values", self.dis_x_pred)

            with tf.variable_scope("generator", reuse=True):
                self.sampler = self._generator(self.z)

            with tf.variable_scope("losses"):
                self.discriminator_loss = self._discriminator_loss()
                self.generator_loss = self._generator_loss()

                train_variables = tf.trainable_variables()
                self.discriminator_vars = [var for var in train_variables if "discriminator" in var.name]
                self.generator_vars = [var for var in train_variables if "generator" in var.name]

                with tf.name_scope("discriminator_optimizer"):
                    self.d_optim = self._adam_optimizer(self.discriminator_loss, self.discriminator_vars, learning_rate)
                with tf.name_scope("generator_optimizer"):
                    self.g_optim = self._adam_optimizer(self.generator_loss, self.generator_vars, learning_rate)

    def _discriminator(self, x):
        x = tf.reshape(x, [self.batch_size, self.image_size, self.image_size, self.image_channels])
        conv1 = nn_ops.conv2d_contrib(x, 64, kernel=4, stride=2, activation_fn=nn_ops.leaky_relu_batch_norm, scope="conv1")
        conv2 = nn_ops.conv2d_contrib(conv1, 128, kernel=4, stride=2, activation_fn=nn_ops.leaky_relu_batch_norm, scope="conv2")
        conv3 = nn_ops.conv2d_contrib(conv2, 256, kernel=4, stride=2, activation_fn=nn_ops.leaky_relu_batch_norm, scope="conv3")
        conv4 = nn_ops.conv2d_contrib(conv3, 512, kernel=4, stride=2, padding="VALID", activation_fn=nn_ops.leaky_relu_batch_norm, scope="conv4")
        conv4 = nn_ops.flatten_contrib(conv4)
        fc = nn_ops.linear_contrib(conv4, 1024, activation_fn=nn_ops.leaky_relu_batch_norm, scope="fully_connected")
        predicted = nn_ops.linear_contrib(fc, 1, activation_fn=tf.nn.sigmoid, scope="prediction")
        return predicted

    def _generator(self, z):
        fc = nn_ops.linear_contrib(z, 4 * 4 * 1024, activation_fn=tf.nn.sigmoid)
        z = tf.reshape(fc, shape=(tf.shape(z)[0], 4, 4, 1024))
        z = nn_ops.leaky_relu_batch_norm(z)
        deconv1 = nn_ops.conv2d_transpose_contrib(z, 512, kernel=4, stride=2, activation_fn=nn_ops.leaky_relu_batch_norm, scope="upconv1")
        deconv2 = nn_ops.conv2d_transpose_contrib(deconv1, 256, kernel=5, stride=2, activation_fn=nn_ops.leaky_relu_batch_norm, scope="upconv2")
        deconv3 = nn_ops.conv2d_transpose_contrib(deconv2, 128, kernel=5, stride=2, activation_fn=nn_ops.leaky_relu_batch_norm, scope="upconv3")
        deconv4 = nn_ops.conv2d_transpose_contrib(deconv3, self.image_channels, kernel=5, stride=2, activation_fn=nn_ops.leaky_relu_batch_norm, scope="upconv4")
        return deconv4

    def _discriminator_loss(self, eps=1e-8):
        with tf.name_scope("discriminator_loss"):
            dis_loss = tf.reduce_mean(-1.0 * tf.log(tf.clip_by_value(self.dis_x, eps, 1.0)) -
                                      tf.log(tf.clip_by_value(1.0 - self.dis_x_pred, eps, 1.0)))
            tf.summary.scalar("discriminator_loss_mean", dis_loss)
            return dis_loss

    def _generator_loss(self, eps=1e-8):
        with tf.name_scope("generator_loss"):
            gen_loss = tf.reduce_mean(-1.0 * tf.log(tf.clip_by_value(self.dis_x_pred, eps, 1.0)))
            tf.summary.scalar("generator_loss_mean", gen_loss)
            return gen_loss

    def _wasserstein_discriminator_loss(self):
        with tf.name_scope("discriminator_loss"):
            dis_loss = tf.reduce_mean(self.dis_x - self.dis_x_pred)
            tf.summary.scalar("discriminator_loss_mean", dis_loss)
            return dis_loss

    def _wasserstein_decoder_loss(self):
        with tf.name_scope("decoder_loss"):
            gen_loss = tf.reduce_mean(self.dis_x_pred)
            tf.summary.scalar("decoder_loss_mean", gen_loss)
            return gen_loss

    def _adam_optimizer(self, loss, loss_params, learning_rate, beta1=0.5):
        optimizer_dis = tf.train.AdamOptimizer(learning_rate, beta1=beta1)
        grads_dis = optimizer_dis.compute_gradients(loss, var_list=loss_params)
        train_dis = optimizer_dis.apply_gradients(grads_dis)
        grad_norms = self._l2_norms(grads_dis)
        tf.summary.histogram("gradient_l2_norms", grad_norms)
        return train_dis

    def _l2_norms(self, gradients):
        return [tf.nn.l2_loss(g) for g, v in gradients if g is not None]

    def update_params(self, sess, input_tensor):
        # Update discriminator network
        _ = sess.run([self.d_optim], feed_dict={self.x: input_tensor})

        # Update generator network
        _ = sess.run([self.g_optim], feed_dict={self.x: input_tensor})

        # Run generator twice to make sure that discriminator loss does not go to zero (different from paper)
        _ = sess.run([self.g_optim], feed_dict={self.x: input_tensor})

        d_loss, g_loss = sess.run([self.discriminator_loss, self.generator_loss], feed_dict={self.x: input_tensor})
        return d_loss, g_loss

    def generate_samples(self, sess, num_samples):
        z = np.random.normal(size=(num_samples, self.z_size))
        samples = sess.run(self.sampler, feed_dict={self.z: z})
        return np.array(samples)

    def initialize_summaries(self, sess, summary_directory):
        self.merged_summary_op = tf.summary.merge_all()
        self.summary_writer = tf.summary.FileWriter(summary_directory, sess.graph)

    def update_summaries(self, sess, x, epoch):
        if self.merged_summary_op is not None:
            summary = sess.run(self.merged_summary_op, feed_dict={self.x: x})
            self.summary_writer.add_summary(summary, global_step=epoch)
            print("updated summaries...")
