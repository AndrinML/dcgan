""" Training """

import os
import tensorflow as tf

from dcgan import DCGAN
from dcgan_vgg import DCGAN_VGG
import hdf5_dataset
from visualizer import ImageVisualizer

flags = tf.app.flags

flags.DEFINE_string("dataset", "datasets/tree_skel_all_6k_400_1v_64x64.h5", "sample data dir")
flags.DEFINE_string("data_dir", "data/", "checkpoint and logging data dir")
flags.DEFINE_integer("batch_size", 128, "batch size")
flags.DEFINE_integer("image_size", 64, "image size")
flags.DEFINE_integer("channels", 1, "color channels")
flags.DEFINE_integer("max_epoch", 500, "max epoch")
flags.DEFINE_integer("z_size", 256, "size of latent (feature?) space")
flags.DEFINE_float("learning_rate", 5e-4, "learning rate")
flags.DEFINE_integer("generation_step", 1, "generate random images")

FLAGS = flags.FLAGS


def main(_):
    # create experiment folder
    experiment_number = 1
    experiment_dir = os.path.join(FLAGS.data_dir, "DCGAN-%02d" % experiment_number)
    while os.path.exists(experiment_dir):
        experiment_number += 1
        experiment_dir = os.path.join(FLAGS.data_dir, "DCGAN-%02d" % experiment_number)
    os.makedirs(experiment_dir)
    print('created directory:', experiment_dir)

    # load training data
    train_data = hdf5_dataset.read_data_set(FLAGS.dataset, image_size=FLAGS.image_size, shape=(FLAGS.image_size, FLAGS.image_size), binarized=False, validation=0).train

    # create a data visualizer
    visualizer = ImageVisualizer(experiment_dir, image_size=FLAGS.image_size)
    visualizer.training_data_sample(train_data)

    # create the actual DCGAN model
    dcgan_model = DCGAN(FLAGS.image_size, FLAGS.channels, z_size=FLAGS.z_size, learning_rate=FLAGS.learning_rate)

    print("start", type(dcgan_model).__name__, "model training")

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        dcgan_model.initialize_summaries(sess, experiment_dir)

        for epoch in range(FLAGS.max_epoch):

            for images in train_data.next_batch(FLAGS.batch_size):
                d_loss, g_loss = dcgan_model.update_params(sess, images)

            print("epoch: %3d" % epoch, "Discriminator loss %.4f" % d_loss, "Generator loss  %.4f" % g_loss)

            dcgan_model.update_summaries(sess, images, epoch)

            if epoch % FLAGS.generation_step == 0:
                visualizer.save_generated_samples(dcgan_model.generate_samples(sess, num_samples=200), epoch)

if __name__ == '__main__':
    tf.app.run()
