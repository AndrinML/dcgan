"""
Andrin Jenal, 2017
ETH Zurich
"""


import os
import numpy as np
from scipy.misc import imsave


class ImageVisualizer:
    def __init__(self, experiment_dir, image_size):
        self.ROWS = 10
        self.COLS = 20
        self.NUM_SAMPLES = self.ROWS * self.COLS

        self.experiment_dir = experiment_dir
        self.image_size = image_size

    def _normalize(self, x):
        return (x - x.min()) * (1.0 / (x.max() - x.min()))

    def training_data_sample(self, train_data, save_name="dataset_samples.png"):
        idxs = np.random.randint(0, train_data.length(), self.NUM_SAMPLES)
        rand_imgs = train_data.images[idxs]
        image_matrix = self._create_image_grid(self._normalize(rand_imgs), self.NUM_SAMPLES, self.image_size, self.image_size, self.ROWS, self.COLS, ypad=2, xpad=2)
        self._save_image_matrix(image_matrix, save_name)

    def save_generated_samples(self, sample_images, epoch):
        # sample reconstruction
        image_matrix = self._create_image_grid(self._normalize(sample_images), sample_images.shape[0], self.image_size, self.image_size, self.ROWS, self.COLS, ypad=2, xpad=2)
        name = "epoch_%03d.png" % epoch
        self._save_image_matrix(image_matrix, name)
        print("generated and saved images...")

    def save_transition_samples(self, sample_images, image_size, rows, cols, name="transition_samples"):
        image_matrix = self._create_image_grid(self._normalize(sample_images), sample_images.shape[0], image_size, image_size, rows, cols, ypad=2, xpad=2)
        self._save_image_matrix(image_matrix, name + ".png")

    def _create_image_grid(self, x, num_samples, image_height, image_width, rows, cols, ypad=0, xpad=0):
        imgs = np.reshape(x, (num_samples, image_height, image_width, -1))
        image_matrix = np.ones((rows * image_height + (1 + rows) * ypad, cols * image_width + (1 + cols) * xpad, imgs.shape[-1]))
        for i in range(0, rows):
            for j in range(0, cols):
                if i * cols + j < num_samples:
                    yoffset = (i+1) * ypad
                    xoffset = (j+1) * xpad
                    image_matrix[yoffset + i*image_height:yoffset + (i+1)*image_height, xoffset + j*image_width:xoffset + (j+1)*image_width] = imgs[i * cols + j][0:image_height][0:image_width]
        return image_matrix

    def _save_image_matrix(self, image_matrix, save_name):
        save_path = os.path.join(self.experiment_dir, save_name)
        imsave(save_path, np.squeeze(image_matrix))
