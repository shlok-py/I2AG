import torch
from torch.utils.data import Dataset
from torchvision import transforms

IMG_SIZE = 64

class CustomDataset(Dataset):
    def __init__(self, images, spectrograms):
        self.images = images
        self.spectrograms = spectrograms
        # self.image_transforms = image_transforms
        
        self.data_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE,IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize
        ])
        # self.compose_data_transform = transforms.Compose(self.data_transforms)
        

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        print('self.image',len(self.images))
        image = self.images[idx]
        spectrogram = self.spectrograms[idx]
        # Convert to torch tensors if not already
        if not isinstance(image, torch.Tensor):
            # image = self.data_transforms(image)
            print(image.shape)
            image = torch.from_numpy(image)
            print(image.shape)
                
        if not isinstance(spectrogram, torch.Tensor):
            spectrogram = torch.from_numpy(spectrogram)
        return image, spectrogram