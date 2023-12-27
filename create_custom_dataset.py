import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
IMG_SIZE = 256

class CustomDataset(Dataset):
    def __init__(self, images, spectrograms):
        self.images = images
        self.spectrograms = spectrograms
        # self.image_transforms = image_transforms
        
        self.data_transforms = transforms.Compose([
                    transforms.ToTensor(),
                    lambda x: x.float(),  # Convert the tensor to torch.float32
                    transforms.Resize((256,256)),
                    transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalize

        ])
        # self.compose_data_transform = transforms.Compose(self.data_transforms)
         
        self.spectrogram_data_transforms = transforms.Compose([
                    ResizeSpectrogram(size=(256, 256)),
                    transforms.Normalize(mean=[0.5], std=[0.5])


        ])
        

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        print('self.image',len(self.images))
        image = self.images[idx]
        spectrogram = self.spectrograms[idx]
        # Convert to torch tensors if not already
        if not isinstance(image, torch.Tensor):
            # image = Image.fromarray(image)
            print("Hello image type",type(image))
            image = self.data_transforms(image)
            # print("\n image\n", image.shape, "\n\n\n")
            image = np.array(image)
            # print(image.shape)
            image = torch.from_numpy(image)
            print(image.shape)
                
        if not isinstance(spectrogram, torch.Tensor):
            print('\nspectrogram\n\n', spectrogram.shape, '\n\n')
            spectrogram = self.spectrogram_data_transforms(spectrogram)
            spectrogram = np.array(spectrogram)
            print('\nspectrogram\n\n', spectrogram.shape, '\n\n')
            
            spectrogram = torch.from_numpy(spectrogram)
        return image, spectrogram
class ResizeSpectrogram:
    def __init__(self, size):
        self.size = size

    def __call__(self, spectrogram):
        # Convert the spectrogram to a PyTorch tensor
        spectrogram_tensor = torch.from_numpy(spectrogram)
        spectrogram_tensor = spectrogram_tensor[:3, :, :]

        # Resize the spectrogram tensor
        resized_spectrogram = torch.nn.functional.interpolate(
            spectrogram_tensor.unsqueeze(0),  # Add a batch dimension
            size=self.size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)  # Remove the batch dimension

        return resized_spectrogram
