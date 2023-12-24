# not working or i could not make it work



import os
import random
import h5py
import numpy as np
from PIL import Image

def link_spectrograms_to_images(spectrograms_folder, images_folder, num_images_per_spectrogram):
    spectrogram_files = [file for file in os.listdir(spectrograms_folder) if file.endswith('.jpg')]
    image_files = [file for file in os.listdir(images_folder) if file.endswith('.jpg')]

    # Dictionary to link spectrograms to multiple images
    linked_data = {}
    for spectrogram_file in spectrogram_files:
        spectrogram_path = os.path.join(spectrograms_folder, spectrogram_file)
        
        # Randomly select num_images_per_spectrogram images for each spectrogram file
        selected_images = random.sample(image_files, min(len(image_files), num_images_per_spectrogram))
        image_paths = [os.path.join(images_folder, img) for img in selected_images]
        linked_data[spectrogram_path] = image_paths

    return linked_data

def dump_linked_data_to_hdf5(linked_data, output_file):
    with h5py.File(output_file, 'w') as hf:
        for i, (spectrogram_path, image_paths) in enumerate(linked_data.items(), start=1):
            try:
                # Check if spectrogram is a NumPy array, if not, convert to a NumPy array
                spectrogram_array = np.array(Image.open(spectrogram_path)) if not isinstance(spectrogram_path, np.ndarray) else spectrogram_path
                # Convert image paths to NumPy arrays (you might want to load images instead)
                image_arrays = [np.array(Image.open(img)) for img in image_paths]
                
                hf.create_dataset(f'spectrogram_{i}', data=spectrogram_array)
                # Create a group for images associated with each spectrogram
                img_group = hf.create_group(f'images_{i}')
                for j, img_array in enumerate(image_arrays, start=1):
                    img_group.create_dataset(f'image_{j}', data=img_array)
            except Exception as e:
                print(f"Error processing data {i}: {e}")

def save_hdf5_to_animal_folder(linked_data, animal_folder):
    output_hdf5_file = os.path.join(animal_folder, 'linkedddd_data.h5')
    dump_linked_data_to_hdf5(linked_data, output_hdf5_file)

animal_folder = "D:\\Animals_processed\\Fox"

spectrograms_folder = os.path.join(animal_folder, 'spectrograms')
images_folder = os.path.join(animal_folder, 'images')

# Number of images to link with each spectrogram
num_images_per_spectrogram = 3  # Adjust as needed

# Linked spectrogram data is created and dumped to HDF5
linked_data = link_spectrograms_to_images(spectrograms_folder, images_folder, num_images_per_spectrogram)
save_hdf5_to_animal_folder(linked_data, animal_folder)
