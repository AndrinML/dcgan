"""
Andrin Jenal, 2017
ETH Zurich
"""

import tensorflow as tf

from dcgan import DCGAN
import hdf5_dataset
from checkpoint_saver import CheckpointSaver
from visualizer import ImageVisualizer

flags = tf.app.flags

flags.DEFINE_string("dataset", "datasets/celeb_dataset_colored.h5", "sample results dir")
flags.DEFINE_string("data_dir", "results/", "checkpoint and logging results dir")
flags.DEFINE_integer("batch_size", 128, "batch size")
flags.DEFINE_integer("image_size", 64, "image size")
flags.DEFINE_integer("channels", 3, "color channels")
flags.DEFINE_integer("max_epoch", 500, "max epoch")
flags.DEFINE_integer("z_size", 256, "size of latent (feature?) space")
flags.DEFINE_float("learning_rate", 5e-4, "learning rate")
flags.DEFINE_integer("generation_step", 1, "generate random images")

FLAGS = flags.FLAGS


def main(_):
    # create checkpoint saver
    # the checkpoint saver, can create checkpoint files, which later can be use to restore a model state, but it also
    # audits the model progress to a log file
    checkpoint_saver = CheckpointSaver(FLAGS.data_dir)
    checkpoint_saver.save_experiment_config(FLAGS.__dict__['__flags'])

    # load training data
    data_set, data_set_shape = hdf5_dataset.read_data_set(FLAGS.dataset, image_size=FLAGS.image_size, shape=(FLAGS.image_size, FLAGS.image_size, FLAGS.channels), binarized=True, validation=0)
    train_data = data_set.train

    # create a data visualizer
    visualizer = ImageVisualizer(checkpoint_saver.get_experiment_dir(), image_size=FLAGS.image_size)
    visualizer.training_data_sample(train_data)

    # create the actual DCGAN model
    dcgan_model = DCGAN(FLAGS.image_size, FLAGS.channels, z_size=FLAGS.z_size, learning_rate=FLAGS.learning_rate)

    print("start", type(dcgan_model).__name__, "model training")
    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        dcgan_model.initialize_summaries(sess, checkpoint_saver.get_experiment_dir())

        for epoch in range(FLAGS.max_epoch):

            for images in train_data.next_batch(FLAGS.batch_size):
                d_loss, g_loss = dcgan_model.update_params(sess, images)

            print("epoch: %3d" % epoch, "Discriminator loss %.4f" % d_loss, "Generator loss  %.4f" % g_loss)

            dcgan_model.update_summaries(sess, images, epoch)

            if epoch % FLAGS.generation_step == 0:
                visualizer.save_generated_samples(dcgan_model.generate_samples(sess, num_samples=200), epoch)

if __name__ == '__main__':
    tf.app.run()
