import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image
import numpy as np
import torchvision.transforms as transforms

class Encoder(nn.Module):
    def __init__(self, input_size, encoding_dim):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.encoding_dim = encoding_dim
        self.encoder = nn.Linear(self.input_size, self.encoding_dim)
        # self.encoder = nn.Sequential(nn.Conv2d(3,16, kernel_size = 3, padding = 1),
        #                             nn.ReLU(),
        #                             nn.MaxPool2d(2,2),
        #                             nn.Conv2d(16, 8, kernel_size = 3, padding = 1),
        #                             nn.ReLU(),
        #                             nn.MaxPool2d(2,2)
        #                             )
       
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(8 * 64 * 64, 128)
    def forward(self, x):
        x = self.flatten(x)
        print("Flatten shape", x.shape)
        x = self.encoder(x)
        # x = self.fc(x)
        
        # x = torch.sigmoid(x)
        return x

if __name__ == "__main__":
    file_path = r"D:\working_data\Ferret\images\0AULGQ4RH5BC.jpg"
    image  = Image.open(file_path)
    # image = image.resize((256,256))
    input_size = 256
    # image_np = np.array(image)
    transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor()
    ])
    image_tensor = transform(image)
    encoder = Encoder(input_size, 128)
    image_embeddings = encoder(image_tensor)
    print(image_embeddings)