import h5py
import os

def concatenate_hdf5_files(input_files, output_file):
    with h5py.File(output_file, 'w') as out_file:
        for input_file in input_files:
            with h5py.File(input_file, 'r') as in_file:
                for key in in_file.keys():
                    # Copy each dataset from the input file to the output file
                    out_file.copy(in_file[key], key)

# List of input HDF5 files
input_hdf5_files = [
    "D:\\30animalsprocessed_audio_folders\\Alligator\\linked_images.h5",
    "D:\\30animalsprocessed_audio_folders\\Bear\\linked_images.h5",
    "D:\\30animalsprocessed_audio_folders\\Cheetah\\linked_images.h5",
    "D:\\30animalsprocessed_audio_folders\\Deer\\linked_images.h5",
    "D:\\30animalsprocessed_audio_folders\\Elephant\\linked_images.h5",
    "D:\\30animalsprocessed_audio_folders\\Fox\\linked_images.h5",
    "D:\\30animalsprocessed_audio_folders\\Frog\\linked_images.h5",
    "D:\\30animalsprocessed_audio_folders\\Gecko\\linked_images.h5",
    "D:\\30animalsprocessed_audio_folders\\Gorilla\\linked_images.h5",
    "D:\\30animalsprocessed_audio_folders\\Hippopotamus\\linked_images.h5",
    "D:\\30animalsprocessed_audio_folders\\Jaguar\\linked_images.h5",
    "D:\\30animalsprocessed_audio_folders\\Leopard\\linked_images.h5",
    "D:\\30animalsprocessed_audio_folders\\Lion\\linked_images.h5",
    "D:\\30animalsprocessed_audio_folders\\Lynx\\linked_images.h5",
    "D:\\30animalsprocessed_audio_folders\\mongoose\\linked_images.h5",
    "D:\\30animalsprocessed_audio_folders\\Monkey\\linked_images.h5",
    "D:\\30animalsprocessed_audio_folders\\moose\\linked_images.h5",
    "D:\\30animalsprocessed_audio_folders\\Orangutan\\linked_images.h5",
    "D:\\30animalsprocessed_audio_folders\\Panda\\linked_images.h5",
    "D:\\30animalsprocessed_audio_folders\\Panther\\linked_images.h5",
    "D:\\30animalsprocessed_audio_folders\\Peacock\\linked_images.h5",
    "D:\\30animalsprocessed_audio_folders\\Raccoon\\linked_images.h5",
    "D:\\30animalsprocessed_audio_folders\\Rhino\\linked_images.h5",
    "D:\\30animalsprocessed_audio_folders\\Snake\\linked_images.h5",
    "D:\\30animalsprocessed_audio_folders\\Wild_Bear\\linked_images.h5",
    "D:\\30animalsprocessed_audio_folders\\Wolf\\linked_images.h5",
    "D:\\30animalsprocessed_audio_folders\\Zebra\\linked_images.h5"
]

# Output file path
output_hdf5_file = "D:\\30animalsprocessed_audio_folders\\combined_linked_images.h5"

# Concatenate the HDF5 files
concatenate_hdf5_files(input_hdf5_files, output_hdf5_file)
