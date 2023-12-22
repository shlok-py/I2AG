import os

import numpy as np

from autoencoder.autoencoder import VAE
from PIL import Image


LEARNING_RATE = 0.0005
BATCH_SIZE = 6
EPOCHS = 10

SPECTROGRAMS_PATH = "/home/shlok/working_data/Frog/spectrograms/"
IMAGES_PATH = "/home/shlok/working_data/Frog/images/"


def load_fsdd(spectrograms_path):
    x_train = []
    for root, _, file_names in os.walk(spectrograms_path):
        for file_name in file_names:
            file_path = os.path.join(root, file_name)
            spectrogram = np.load(file_path) # (n_bins, n_frames, 1)
            x_train.append(spectrogram)
    x_train = np.array(x_train)
    x_train = x_train[..., np.newaxis] # -> (3000, 256, 64, 1)
    return x_train

def save_load_images(images_path):
    x_train = []
    for root, _, file_names in os.walk(images_path):
        for file_name in file_names:
            file_path = os.path.join(root, file_name)
            # print("file_path",file_path)
            image = Image.open(file_path)
            # image = np.array(image)
            x_train.append(image)
    x_train = np.array(x_train)
    x_train = x_train[..., np.newaxis] # -> (3000, 256, 64, 1)
    return x_train


def train(x_train, y_train, learning_rate, batch_size, epochs):
    autoencoder = VAE(
        input_shape=(256, 173, 1),
        conv_filters=(512, 256, 128, 64, 32),
        conv_kernels=(3, 3, 3, 3, 3),
        conv_strides=(2, 2, 2, 2, (2, 1)),
        latent_space_dim=128
    )
    autoencoder.summary()
    autoencoder.compile(learning_rate)
    autoencoder.train(x_train, y_train, batch_size, epochs)
    return autoencoder


if __name__ == "__main__":
    x_train = save_load_images(IMAGES_PATH)
    # y_train = load_fsdd(SPECTROGRAMS_PATH)
    # autoencoder = train(x_train, y_train, LEARNING_RATE, BATCH_SIZE, EPOCHS)
    # autoencoder.save("model")
    print(x_train.shape)