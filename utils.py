# from torchvision import transforms
import torch
import torchvision
from torch.utils.data import DataLoader
import numpy as np
import h5py
import matplotlib.pyplot as plt
from create_custom_dataset import CustomDataset
from PIL import Image
from tqdm import tqdm
IMG_SIZE = 64
BATCH_SIZE = 4

# def get_data(file_path):
#     Image = []
#     Spectrograms = []
#     with h5py.File(file_path, 'r') as hf:
#         for i in hf.keys():
#             if 'image' in i.lower():
#                 Image.append(hf[i][:])
#                 continue
#             if 'spectrogram' in i.lower():
#                 Spectrograms.append(hf[i][:])
#                 continue
#         print("Image array:", Image)
#         print("Spectrogram array:", Spectrograms)
        
#         print(len(Image), len(Spectrograms))
#         with h5py.File('new_data.h5', 'w') as nhf:
#             # nhf.create_group()
#             for i in range(len(Image)):
#                 group = nhf.create_group(f"image{i}")
                
#                 group.create_dataset('image', data = Image[i])
#                 group.create_dataset('spectrogram', data =Spectrograms[i])
#             nhf.close()
                
#     hf.close()
#     print("Done")

def get_data(file_path):
    with h5py.File(file_path, 'r') as hf:
        # print(hf.keys())
        image = []
        spectrogram = []
        for group_name in hf.keys():
            group = hf[group_name]
            
            # Read the image and spectrogram datasets
            image_data = group['image'][:]
            spectrogram_data = group['spectrogram'][:]
            # print("image", image_data.shape, "\n spectrogram", spectrogram_data.shape)
            # plt.subplot(1,2,1)
            # plt.imshow(image_data)
            # plt.subplot(1,2,2)
            # plt.imshow(spectrogram_data)
            # plt.show()
            # break
            # print(spectrogram_data.shape)
            spectrogram_data = np.transpose(spectrogram_data, (2, 0, 1))
            spectrogram_data = np.pad(spectrogram_data, ((0, 0), (0, 600), (0, 0)), mode='constant')
            # print("initial spectrogram data", spectrogram_data.shape)
            image_data = image_data.astype(np.float32)
            spectrogram_data = spectrogram_data.astype(np.float32)
            image.append(image_data)
            spectrogram.append(spectrogram_data)
        print('image', type(image[0]))
        return image, spectrogram
# def load_data():
    # image, spec = get_data('new_data.h5')
    # dataset = CustomDataset(image, spec)
    # batch_size = 2
    # dataloader = DataLoader(dataset, batch_size=batch_size)
    # print(dataloader)
    # for batch in dataloader:
    #     images_batch, spectrograms_batch = batch
    #     print('images_batch',images_batch) 
    #     print('\n spectrograms_batch',spectrograms_batch)
    #     break
    # print(dataset[0])


def load_transformed_dataset():
    image, spec = get_data('../new_data.h5')
    # print(len(image))
    # image = np.asarray(image)
    # spec = np.asarray(spec)
    # print(type(image), type(spec))
    dataset = CustomDataset(image, spec)
    # print(dataset[0][0])
    # plt.imshow(np.asarray(dataset[0][0]))
    # plt.show()
    # print(dataset[0][0].shape)
    # data_transform = transforms.Compose(data_transforms)
    batch_size = 2
    dataloader = DataLoader(dataset, batch_size=batch_size)
    return dataloader

def plot_images(images):
    plt.figure(figsize=(32, 32))
    plt.imshow(torch.cat([
        torch.cat([i for i in images.cpu()], dim=-1),
    ], dim=-2).permute(1, 2, 0).cpu())
    plt.show()


def save_images(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(ndarr)
    im.save(path)
    
if __name__ == '__main__':
    # get_data("../../data.h5")
    # get_data('new_data.h5')
    # load_data()
    # load_transformed_dataset()
    pbar = tqdm(load_transformed_dataset())
    # print('\npbar', pbar)
    # print('')
    for i, (images, spectrograms) in enumerate(pbar):
        print(i, '\t image \t', images)
        print(i , '\t spectrograms \t', spectrograms)
    # get_data('../new_data.h5')