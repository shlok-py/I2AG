import os
import h5py

def combine_hdf5_files(animal_folders, output_combined_file):
    with h5py.File(output_combined_file, 'w') as combined_hf:
        for animal_folder in animal_folders:
            animal_name = os.path.basename(animal_folder)
            animal_hdf5_file = os.path.join(animal_folder, f'linked_images.h5')

            with h5py.File(animal_hdf5_file, 'r') as animal_hf:
                # Copy each dataset from the animal HDF5 file to the combined HDF5 file
                for key in animal_hf.keys():
                    combined_hf.copy(animal_hf[key], f'{animal_name}_{key}')

# Set the paths to animal folders
animal_folders = ["D:\\30animalsprocessed_audio_folders\\Alligator", "D:\\30animalsprocessed_audio_folders\\Bear","D:\\30animalsprocessed_audio_folders\\Cheetah",
                  "D:\\30animalsprocessed_audio_folders\\Deer","D:\\30animalsprocessed_audio_folders\\Elephant","D:\\30animalsprocessed_audio_folders\\Fox",
                  "D:\\30animalsprocessed_audio_folders\\Frog","D:\\30animalsprocessed_audio_folders\\Lion","D:\\30animalsprocessed_audio_folders\\Gorilla",
]

output_combined_file = "D:\\30animalsprocessed_audio_folders\\Allfiles_in_one_dumped.h5"

# Combine separate HDF5 files into a single HDF5 file
combine_hdf5_files(animal_folders, output_combined_file)
