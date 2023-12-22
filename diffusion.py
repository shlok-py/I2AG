import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from torchvision.utils import make_grid
from PIL import Image
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# image_path = '/home/shlok/working_data/Ferret/images/0BPYR995WNU5.jpg'
image_path = 'd://working_data/Ferret/images/0BPYR995WNU5.jpg'
class Diffusion:
    def __init__(self, noise_steps = 1000, beta_start = 0.0001, beta_end = 0.02, image_size = 256, device = 'cpu') -> None:
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.device = device
        self.image_size = image_size
        self.beta = self.get_beta(self.beta_start, self.beta_end).to(device) #get linear variance schedule
        self.alpha = 1. - self.beta
        # self.sqrt_alpha = torch.sqrt(self.alpha)
        self.alpha_hat = self._get_alpha_hat(self.alpha) # get the cumulative product of alpha
        self.sqrt_alpha_hat = torch.sqrt(self.alpha_hat)
        # self.sqrt_alpha_inverse = torch.sqrt(1-self.alpha)
        self.sqrt_alpha_hat_inverse = torch.sqrt(1-self.alpha_hat)
        # print(self.sqrt_alpha)
        self.noise = self._get_gaussian_noise()
    
    def _get_alpha_hat(self, alpha):
        # print(torch.cumprod(self.alpha, dim = 0))
        return torch.cumprod(self.alpha, dim = 0)
    
    def get_beta(self, beta_start, beta_end):
        betas = torch.linspace(beta_start, beta_end, self.noise_steps).to(self.device)
        return betas
    def _get_gaussian_noise(self):
        return torch.randn(3, self.image_size, self.image_size).to(self.device)
    
    def t_noiser(self, x, t):
        # noise = self._get_gaussian_noise() 
        return self.sqrt_alpha_hat[t]*x + self.sqrt_alpha_hat_inverse[t] * self.noise, self.noise
    
    def noiser(self, x): # noise the image
        noised = []
        for i in range(self.noise_steps):
            x = self._noiser(i,x)
            if i%100 == 0:
                noised.append(x)
        return x, noised
        # return self.sqrt_alpha *x + self.sqrt_alpha_inverse * self._get_gaussian_noise()
    def _noiser(self, t,x):
        return self.sqrt_alpha[t] *x + self.sqrt_alpha_inverse[t] * self._get_gaussian_noise()

if __name__ == "__main__":
    diffusion = Diffusion()
    x = Image.open(image_path)
    resize = transforms.Resize((256,256))
    x = resize(x)
    plt.imshow(x)
    plt.show()
    x = transforms.ToTensor()(x)
    x = x.to(diffusion.device)
    print(x.shape)
    # x = x.to(diffusion.device)
    # x_noisy, noised = diffusion.noiser(x)
    x_noisy, noise = diffusion.t_noiser(x, 199)
    print(x_noisy.shape) 
    # plt.subplot(1,2,1)
    # x = np.transpose(x, (1, 2, 0))
    # plt.imshow(x)
    # image = np.transpose(x_noisy, (1, 2, 0))
    # plt.subplot(1,2,2)
    # plt.imshow(image)
    # print(len``)
    # for i in range(len(noised)):
    #     plt.subplot(2,5,i+1)
    #     image = noised[i]
    #     x = np.transpose(image, (1, 2, 0))

    #     plt.imshow(x)

    x = np.transpose(x_noisy, (1, 2, 0))
    plt.imshow(x)

    plt.show()     
    # x = x.squeeze(0)
    # x = x.permute(1,2,0)
    # x = x.cpu().detach().numpy()
    # plt.imshow(x)
    # plt.show()
    # plt.imsave('test.png',x)