import os
import random
import h5py
import numpy as np
from PIL import Image

def link_images_to_spectrograms(images_folder, spectrograms_folder):
    image_files = [file for file in os.listdir(images_folder) if file.endswith('.jpg')]
    spectrogram_files = [file for file in os.listdir(spectrograms_folder) if file.endswith('.jpg')]

    # Shuffling
    random.shuffle(image_files)
    # random.shuffle(spectrogram_files)

    # dictionary to link images to spectrograms
    linked_images = {}
    for i in range(min(len(image_files), len(spectrogram_files))):
        image_path = os.path.join(images_folder, image_files[i])
        spectrogram_path = os.path.join(spectrograms_folder, spectrogram_files[i])
        linked_images[image_path] = spectrogram_path

    return linked_images

def dump_linked_images_to_hdf5(linked_images, output_file):
    with h5py.File(output_file, 'w') as hf:
        for i, (image_path, spectrogram_path) in enumerate(linked_images.items(), start=1):

            # Check if images are NumPy arrays, if not, convert to NumPy arrays
            image_array = np.array(Image.open(image_path)).astype(float)
            spectrogram_array = np.array(Image.open(spectrogram_path)).astype(float)

            group = hf.create_group(f"image{i}")
            group.create_dataset('image', data=image_array)
            group.create_dataset('spectrogram', data=spectrogram_array)

def save_hdf5_to_animal_folder(linked_images, animal_folder):
    output_hdf5_file = os.path.join(animal_folder, 'linked_images.h5')
    dump_linked_images_to_hdf5(linked_images, output_hdf5_file)

animal_folder ="D:\\30animalsprocessed_audio_folders\\Zebra"


images_folder = os.path.join(animal_folder, 'images')
spectrograms_folder = os.path.join(animal_folder, 'spectrograms')


#  linked images named folder is created  and dumped to HDF5
linked_images = link_images_to_spectrograms(images_folder, spectrograms_folder)

save_hdf5_to_animal_folder(linked_images, animal_folder)
