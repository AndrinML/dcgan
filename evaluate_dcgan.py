"""
Andrin Jenal, 2017
ETH Zurich
"""

import os
import argparse
import json
import tensorflow as tf
import numpy as np

from dcgan import DCGAN
from visualizer import ImageVisualizer


def dcgan_image_transition(z, checkpoint_file, model_params):
    tree_net = DCGAN(**model_params)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, checkpoint_file)
        x_transition = sess.run(tree_net.sampler, feed_dict={tree_net.z: z})
    return x_transition


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint_file_path', help='model to be restored')
    parser.add_argument('config_file', help='you guess it')
    args = parser.parse_args()

    # restore experiment parameters
    with open(args.config_file, 'r') as jfile:
        model_params = json.load(jfile)

    image_height = model_params["image_height"]
    image_width = model_params["image_width"]
    file_name = os.path.splitext(os.path.basename(args.image_path))[0]

    num_steps = 10
    z_range = np.random.uniform(-1, 1, 2)
    random_z = np.linspace(z_range[0], z_range[1], num_steps)
    x_transition = dcgan_image_transition(random_z, args.checkpoint_file_path, model_params)

    visualizer = ImageVisualizer("samples", image_height)
    visualizer.save_transition_samples(x_transition, image_height, 1, num_steps, name="random_transition")
