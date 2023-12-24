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
    random.shuffle(spectrogram_files)

    # Lists to store linked arrays
    linked_images = []
    linked_spectrograms = []

    for i in range(min(len(image_files), len(spectrogram_files))):
        image_path = os.path.join(images_folder, image_files[i])
        spectrogram_path = os.path.join(spectrograms_folder, spectrogram_files[i])

        # Check if images are NumPy arrays, if not, convert to NumPy arrays
        image_array = np.array(Image.open(image_path)) if not isinstance(image_path, np.ndarray) else image_path
        spectrogram_array = np.array(Image.open(spectrogram_path)) if not isinstance(spectrogram_path, np.ndarray) else spectrogram_path

        linked_images.append(image_array)
        linked_spectrograms.append(spectrogram_array)

    return linked_images, linked_spectrograms

def dump_linked_arrays_to_hdf5(linked_images, linked_spectrograms, output_file):
    with h5py.File(output_file, 'w') as hf:
        for i, (image_array, spectrogram_array) in enumerate(zip(linked_images, linked_spectrograms), start=1):
            hf.create_dataset(f'image_{i}', data=image_array)
            hf.create_dataset(f'spectrogram_{i}', data=spectrogram_array)

def combine_linked_arrays(animal_folders, output_combined_file):
    all_linked_images = []
    all_linked_spectrograms = []

    for animal_folder in animal_folders:
        images_folder = os.path.join(animal_folder, 'images')
        spectrograms_folder = os.path.join(animal_folder, 'spectrograms')

        linked_images, linked_spectrograms = link_images_to_spectrograms(images_folder, spectrograms_folder)

        all_linked_images.extend(linked_images)
        all_linked_spectrograms.extend(linked_spectrograms)

    dump_linked_arrays_to_hdf5(all_linked_images, all_linked_spectrograms, output_combined_file)

# Example Usage
animal_folders = ["D:\\Animals_processed\\Fox", "D:\\Animals_processed\\Elephant", "D:\\Animals_processed\\Cheetah","D:\\Animals_processed\\Bear","D:\\Animals_processed\\Deer"]
output_combined_file = "D:\\Animals_processed\\NewLinkedArrays.h5"

combine_linked_arrays(animal_folders, output_combined_file)
