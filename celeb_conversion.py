"""
Andrin Jenal, 2017
ETH Zurich
"""

import os
import h5py
from zipfile import ZipFile
import numpy as np
from scipy import misc


class CelebDataset:
    def __init__(self, dataset_destination_dir, image_size=64, channels=1):
        self.dataset_dir = dataset_destination_dir
        self.image_size = image_size
        self.channels = channels

    def create_dataset_from_zip(self, path_to_zip_file, num_images=None, dataset_filename="celeb_dataset.h5"):
        images = []
        image_names = []
        with ZipFile(path_to_zip_file, 'r') as zfile:
            file_list = zfile.namelist()

            for ith, img_file in enumerate(file_list):
                if str(img_file).endswith('.jpg'):
                    with zfile.open(img_file) as imf:
                        img = misc.imread(imf, mode="RGB")
                        image = self.get_normalized_image(img, self.image_size, self.image_size)
                        if self.channels == 1:
                            image = self.image2gray(image)
                        images.append(image)
                        image_names.append(img_file)
                        if ith == num_images:
                            break

        file_name_path = os.path.join(self.dataset_dir, dataset_filename)
        with h5py.File(file_name_path, 'a') as hfile:
            self.save_images_to_hdf5(hfile, zip(image_names, images))

    def resize_width(self, image, width=64.):
        h, w = np.shape(image)[:2]
        return misc.imresize(image, [int((float(h) / w) * width), width])

    def center_crop(self, x, height=64):
        h = np.shape(x)[0]
        j = int(round((h - height) / 2.))
        return x[j:j + height, :, :]

    def get_normalized_image(self, img, width=64, height=64):
        return self.center_crop(self.resize_width(img, width=width), height=height)

    def save_images_to_hdf5(self, open_h5file, image_list):
        for img_name, img_data in image_list:
            dataset = open_h5file.create_dataset(self.get_filename(img_name), data=img_data, shape=img_data.shape)

    def get_filename(self, path):
        return os.path.splitext(os.path.basename(path))[0]

    def image2gray(self, image):
        return image[:, :, 0] * 0.299 + image[:, :, 1] * 0.587 + image[:, :, 2] * 0.114

if __name__ == '__main__':
    c = CelebDataset("/home/ajenal/NeuralNetworks/dcgan-vgg/datasets/", image_size=64, channels=3)
    c.create_dataset_from_zip("/home/ajenal/Downloads/img_align_celeba.zip", 3000, dataset_filename="celeb_dataset_colored.h5")
