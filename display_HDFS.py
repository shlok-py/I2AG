import h5py
import numpy as np
import matplotlib.pyplot as plt

# Path to your HDF5 file
hdf5_file_path = "D:\\Animals_processed\\Elephant\\linked_images.h5"

with h5py.File(hdf5_file_path, 'r') as hf:
    # Access linked datasets 
    num_samples = min(10, len(hf.keys()))  # Display 10 samples
    sample_indices = np.random.choice(range(len(hf.keys())), num_samples, replace=False)

    # Display linked spectrogram-image pairs
    for i, index in enumerate(sample_indices, start=1):
        image_dataset_name = f'image_{index + 1}'
        spectrogram_dataset_name = f'spectrogram_{index + 1}'

        # Check if the dataset names exist before accessing
        if image_dataset_name in hf and spectrogram_dataset_name in hf:
            image_data = hf[image_dataset_name][:]
            spectrogram_data = hf[spectrogram_dataset_name][:]

            # Display the linked pair
            plt.figure(figsize=(8, 4))
            plt.subplot(1, 2, 1)
            plt.imshow(image_data)
            plt.title(f'Image - {image_dataset_name}')

            plt.subplot(1, 2, 2)
            plt.imshow(spectrogram_data)
            plt.title(f'Spectrogram - {spectrogram_dataset_name}')

            plt.show()
        else:
            print(f"Dataset not found: {image_dataset_name} or {spectrogram_dataset_name}")
