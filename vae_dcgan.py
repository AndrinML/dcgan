"""
Andrin Jenal, 2017
ETH Zurich
"""


import tensorflow as tf
import numpy as np

import nn_ops


class VAE_DCGAN:

    def __init__(self, image_size, channels, z_size=256, learning_rate_enc=5e-4, learning_rate_dis=5e-4, learning_rate_dec=5e-4):
        # summaries
        self.merged_summary_op = None
        self.summary_writer = None

        self.image_size = image_size
        self.image_channels = channels
        self.z_size = z_size
        self.learning_rate_enc = learning_rate_enc
        self.learning_rate_gen = learning_rate_dec
        self.learning_rate_dis = learning_rate_dis
        self.eps = 1e-8

        self.x = tf.placeholder(tf.float32, shape=(None, self.image_size, self.image_size, self.image_channels))
        self.batch_size = tf.shape(self.x)[0]

        self.z_p = tf.random_normal((self.batch_size, self.z_size), mean=0, stddev=1)
        self.eps = tf.random_normal((self.batch_size, self.z_size), mean=0, stddev=1)

        with tf.variable_scope("vae_dcgan_model"):

            with tf.variable_scope("encoder"):
                self.z_x_mean, self.z_x_log_sigma_sq = self._encoder(self.x)

            with tf.variable_scope("generator"):
                self.z_x = tf.add(self.z_x_mean, tf.multiply(tf.sqrt(tf.exp(self.z_x_log_sigma_sq)), self.eps))
                self.x_tilde = self._generator(self.z_x)

            with tf.variable_scope("discriminator"):
                _, self.l_x_tilde = self._discriminator(self.x_tilde)

            with tf.variable_scope("generator", reuse=True):
                self.x_p = self._generator(self.z_p)

            with tf.variable_scope("discriminator", reuse=True):
                self.dis_x, self.l_x = self._discriminator(self.x)
                tf.summary.histogram("predicted_x_values", self.dis_x)

            with tf.variable_scope("discriminator", reuse=True):
                self.dix_x_p, _ = self._discriminator(self.x_p)
                tf.summary.histogram("predicted_x_p_values", self.dix_x_p)

            with tf.variable_scope("losses"):
                self.prior = self._kl_divergence()

                self.discriminator_loss = self._discriminator_loss()
                self.generator_loss = self._generator_loss()
                self.lth_layer_loss = self._lth_layer_loss()

                self.loss_encoder = self.prior + self.lth_layer_loss
                self.loss_generator = self.lth_layer_loss + self.generator_loss
                self.loss_discriminator = self.discriminator_loss

                train_variables = tf.trainable_variables()
                self.encoder_vars = [var for var in train_variables if "encoder" in var.name]
                self.discriminator_vars = [var for var in train_variables if "discriminator" in var.name]
                self.generator_vars = [var for var in train_variables if "generator" in var.name]

                with tf.name_scope("encoder_optimizer"):
                    self.e_optim = self._adam_optimizer(self.loss_encoder, self.encoder_vars, self.learning_rate_enc)

                with tf.name_scope("discriminator_optimizer"):
                    self.d_optim = self._adam_optimizer(self.loss_discriminator, self.discriminator_vars, self.learning_rate_dis)

                with tf.name_scope("generator_optimizer"):
                    self.g_optim = self._adam_optimizer(self.loss_generator, self.generator_vars, self.learning_rate_gen)

        # initialize saver
        self.saver = tf.train.Saver([v for v in tf.global_variables() if "vae_dcgan_model" in v.name])

        self._check_tensors()

    def _encoder2(self, x):
        x = tf.reshape(x, [self.batch_size, self.image_size * self.image_size * self.image_channels])
        z_mean = nn_ops.linear_contrib(x, self.z_size, activation_fn=None, scope="fully_connected")
        z_log_sigma_sq = nn_ops.linear_contrib(x, self.z_size, activation_fn=None, scope="fully_connected")
        return z_mean, z_log_sigma_sq

    def _encoder(self, x):
        x = tf.reshape(x, [self.batch_size, self.image_size, self.image_size, self.image_channels])
        conv1 = nn_ops.conv2d_contrib(x, 64, kernel=5, stride=2, activation_fn=nn_ops.relu_batch_norm, scope="conv1")
        conv2 = nn_ops.conv2d_contrib(conv1, 128, kernel=3, stride=2, activation_fn=nn_ops.relu_batch_norm, scope="conv2")
        conv3 = nn_ops.conv2d_contrib(conv2, 256, kernel=3, stride=2, padding="VALID", activation_fn=nn_ops.relu_batch_norm, scope="conv3")
        flatten = nn_ops.flatten_contrib(conv3)
        z_mean = nn_ops.linear_contrib(flatten, self.z_size, activation_fn=None, scope="fully_connected")
        z_log_sigma_sq = nn_ops.linear_contrib(flatten, self.z_size, activation_fn=None, scope="fully_connected")
        return z_mean, z_log_sigma_sq

    def _discriminator(self, x):
        x = tf.reshape(x, [self.batch_size, self.image_size, self.image_size, self.image_channels])
        conv1 = nn_ops.conv2d_contrib(x, 64, kernel=5, stride=2, activation_fn=nn_ops.leaky_relu_batch_norm, scope="conv1")
        conv2 = nn_ops.conv2d_contrib(conv1, 128, kernel=3, stride=2, activation_fn=nn_ops.leaky_relu_batch_norm, scope="conv2")
        conv3 = nn_ops.conv2d_contrib(conv2, 256, kernel=3, stride=2, activation_fn=nn_ops.leaky_relu_batch_norm, scope="conv3")
        conv4 = nn_ops.conv2d_contrib(conv3, 256, kernel=3, stride=2, padding="VALID", activation_fn=None, scope="conv4")
        fc = nn_ops.linear_contrib(conv4, 512, activation_fn=None, scope="fully_connected")
        predicted = nn_ops.linear_contrib(fc, 1, activation_fn=tf.nn.sigmoid, scope="prediction")
        net = {"conv1": conv1, "conv2": conv2, "conv3": conv3, "conv4": conv4}
        return predicted, net["conv3"]

    def _generator(self, z):
        fc = nn_ops.linear_contrib(z, 4 * 4 * 1024, activation_fn=None)
        z = tf.reshape(fc, shape=(tf.shape(z)[0], 4, 4, 1024))
        deconv1 = nn_ops.conv2d_transpose_contrib(z, 512, kernel=4, stride=2, activation_fn=tf.nn.relu, scope="upconv1")
        deconv2 = nn_ops.conv2d_transpose_contrib(deconv1, 256, kernel=5, stride=2, activation_fn=tf.nn.relu, scope="upconv2")
        deconv3 = nn_ops.conv2d_transpose_contrib(deconv2, 128, kernel=5, stride=2, activation_fn=tf.nn.relu, scope="upconv3")
        deconv4 = nn_ops.conv2d_transpose_contrib(deconv3, self.image_channels, kernel=5, stride=2, activation_fn=None, scope="upconv4")
        return tf.nn.tanh(deconv4)

    def _discriminator_loss(self, eps=1e-8):
        with tf.name_scope("discriminator_loss"):
            dis_loss = tf.reduce_mean(-1.0 * tf.log(tf.clip_by_value(self.dis_x, eps, 1.0)) -
                                      tf.log(tf.clip_by_value(1.0 - self.dix_x_p, eps, 1.0)))
            tf.summary.scalar("discriminator_loss_mean", dis_loss)
            return dis_loss

    def _generator_loss(self, eps=1e-8):
        with tf.name_scope("generator_loss"):
            gen_loss = tf.reduce_mean(-1.0 * tf.log(tf.clip_by_value(self.dix_x_p, eps, 1.0)))
            tf.summary.scalar("generator_loss_mean", gen_loss)
            return gen_loss

    def _kl_divergence(self):
        with tf.name_scope("kl_divergence_loss"):
            KL = (-0.5 * tf.reduce_sum(1 + tf.clip_by_value(self.z_x_log_sigma_sq, -10.0, 10.0) -
                                                      tf.square(tf.clip_by_value(self.z_x_mean, -10.0, 10.0)) -
                                                      tf.exp(tf.clip_by_value(self.z_x_log_sigma_sq, -10.0, 10.0)), 1)) / (self.image_size * self.image_size)
            KL_mean = tf.reduce_mean(KL)
            tf.summary.histogram("KL_divergence", KL)
            tf.summary.scalar("kl_divergence_mean", KL_mean)
            return KL_mean

    def _lth_layer_loss(self):
        with tf.name_scope("lth_layer_loss"):
            lth_layer_loss = tf.nn.l2_loss((self.l_x - self.l_x_tilde)) / (self.image_size * self.image_size * self.image_channels)
            tf.summary.scalar("lth_layer_loss_mean", lth_layer_loss)
            return lth_layer_loss

    def _wasserstein_discriminator_loss(self):
        with tf.name_scope("discriminator_loss"):
            dis_loss = tf.reduce_mean(self.dis_x - self.dix_x_p)
            tf.summary.scalar("discriminator_loss_mean", dis_loss)
            return dis_loss

    def _wasserstein_decoder_loss(self):
        with tf.name_scope("decoder_loss"):
            gen_loss = tf.reduce_mean(self.dix_x_p)
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

    def _check_tensors(self):
        if tf.trainable_variables():
            for v in tf.trainable_variables():
                print("%s : %s" % (v.name, v.get_shape()))

    def update_params(self, sess, input_tensor):
        _, _, _ = sess.run([self.e_optim, self.g_optim, self.d_optim], feed_dict={self.x: input_tensor})

        kl, d_loss, g_loss, lth_layer = sess.run([self.prior, self.discriminator_loss, self.generator_loss, self.lth_layer_loss], feed_dict={self.x: input_tensor})
        return kl, d_loss, g_loss, lth_layer

    def generate_samples(self, sess, num_samples):
        z = np.random.normal(size=(num_samples, self.z_size))
        samples = sess.run(self.x_p, feed_dict={self.z_p: z})
        return np.array(samples)

    def initialize_summaries(self, sess, summary_directory):
        self.merged_summary_op = tf.summary.merge_all()
        self.summary_writer = tf.summary.FileWriter(summary_directory, sess.graph)

    def update_summaries(self, sess, x, epoch):
        if self.merged_summary_op is not None:
            summary = sess.run(self.merged_summary_op, feed_dict={self.x: x})
            self.summary_writer.add_summary(summary, global_step=epoch)
            print("updated summaries...")

    def restore_model(self, sess, checkpoint_file):
        self.saver.restore(sess, checkpoint_file)
        print("model restored from:", checkpoint_file)
