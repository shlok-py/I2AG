import h5py
import matplotlib.pyplot as plt

# Open the HDF5 file
h5f = h5py.File('data.h5', 'r')

# Open the file as read only
with h5py.File('data.h5', 'r') as hf:
    print(hf.keys())
    # Read the data
    # print(hf.keys())
    # print(hf['Bear_image_102'])
    # for i,j in enumerate(hf.keys()):
    #     # print(i,j)
    #     print(j, hf[j][:])
    #     image = hf[j][:]
    #     plt.imshow(image)
    #     plt.show()
    #     if i ==1:
    #         break
        
    # spec= hf['Bear_spectrogram_102'][:]
    # plt.imshow(spec)
    # plt.show()
    # if 'Fox' in hf.keys():
    #     print('yes')