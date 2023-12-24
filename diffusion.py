import torch
from torch import optim
import torch.nn as nn
import copy
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from torchvision.utils import make_grid
from PIL import Image
from tqdm import tqdm
import os
from UNet import EMA, UNet_conditional
import logging
# from torch.utils.tensorboard import SummaryWriter
from utils import load_transformed_dataset, plot_images, save_images
from encoder import Encoder

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
        self.alpha = 1 - self.beta
        self.sqrt_alpha = torch.sqrt(self.alpha)
        self.alpha_hat = self._get_alpha_hat(self.alpha) # get the cumulative product of alpha
        self.sqrt_alpha_hat = torch.sqrt(self.alpha_hat)
        self.sqrt_alpha_inverse = torch.sqrt(1-self.alpha)
        # print(self.sqrt_alpha)
        # self.noise = self._get_gaussian_noise()
    
    def _get_alpha_hat(self, alpha):
        # print(torch.cumprod(self.alpha, dim = 0))
        return torch.cumprod(self.alpha, dim = 0)
    
    def get_beta(self, beta_start, beta_end):
        betas = torch.linspace(beta_start, beta_end, self.noise_steps).to(self.device)
        return betas
    def _get_gaussian_noise(self,x):
        return torch.randn_like(x).to(self.device)
    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))
    
    def t_noiser(self, x,t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        noise = self._get_gaussian_noise(x)
        # return self.sqrt_alpha_hat[t]*x + self.sqrt_alpha_inverse[t] * self._get_gaussian_noise()
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * noise, noise
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
    
    def sample_spectrograms(self, model, n, image_embeddings):
        print(f"Sampling{n} spectrograms")
        model.eval()
        with torch.no_grad():
            x = torch.randn((n,3,self.image_size, self.image_size)).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position = 0):
                t = (torch.ones(n) * i).long().to(self.device)
                # x_t, noise = diffusion.noise_images(images, t)
                predicted_noise = model(x, t, image_embeddings)
                # loss = mse(noise, predicted_noise)
                alpha = self.alpha[t]
                alpha_hat = self.alpha_hat[t]
                beta = self.beta[t]
                
                if i>1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1/(torch.sqrt(alpha)) * (x-((1-alpha)/torch.sqrt(1-alpha_hat)) * predicted_noise) + torch.sqrt(beta) * noise
            model.train()
            x = (x.clamp(-1,1) + 1) /2
            x = (x * 255)/type(torch.uint8)
            return x
def train(args):
    device = args.device
    dataloader = load_transformed_dataset()
    model = UNet_conditional().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    diffusion = Diffusion(image_size=args.img_size, device=device)
    # logger = SummaryWriter(os.path.join("runs", args.run_name))
    l = len(dataloader)
    ema = EMA(0.995)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)

    for epoch in range(100):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)
        for i, (images, spectrograms) in enumerate(pbar):
            images = images.to(device)
            spectrograms = spectrograms.to(device)
            t = diffusion.sample_timesteps(spectrograms.shape[0]).to(device)
            x_t, noise = diffusion.t_noiser(spectrograms, t)
            # if np.random.random() < 0.1:
            #     labels = None
            print('Images Shape', images.shape)
            encoder = Encoder(torch.prod(torch.tensor(images.shape[-3:])), 256)
            latent_image = encoder(images)
            predicted_noise = model(x_t, t, latent_image)
            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.step_ema(ema_model, model)

            pbar.set_postfix(MSE=loss.item())
            # logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)

        if epoch % 10 == 0:
            # labels = torch.arange(10).long().to(device)
            sampled_images = diffusion.sample_spectrograms(model, n=len(t), image_embeddings=latent_image)
            # ema_sampled_images = diffusion.sample_spectrograms(ema_model, n=len(labels), labels=labels)
            plot_images(sampled_images)
            save_images(sampled_images, os.path.join("results", args.run_name, f"{epoch}.jpg"))
            # save_images(ema_sampled_images, os.path.join("results", args.run_name, f"{epoch}_ema.jpg"))
            torch.save(model.state_dict(), os.path.join("models", args.run_name, f"ckpt{epoch%10}.pt"))
            # torch.save(ema_model.state_dict(), os.path.join("models", args.run_name, f"ema_ckpt.pt"))
            torch.save(optimizer.state_dict(), os.path.join("models", args.run_name, f"optim{epoch%10}.pt"))

def launch():
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "I2AG__"
    args.epochs = 300
    args.batch_size = 2
    args.img_size = 256
    # args.num_classes = 10
    # args.dataset_path = r"C:\Users\dome\datasets\cifar10\cifar10-64\train"
    args.device = "cpu"
    args.lr = 3e-4
    train(args)
    
            
if __name__ == "__main__":
    launch()
    # diffusion = Diffusion()
    # x = Image.open(image_path)
    # resize = transforms.Resize((256,256))
    # x = resize(x)
    # plt.imshow(x)
    # plt.show()
    # x = transforms.ToTensor()(x)
    # x = x.to(diffusion.device)
    # print(x.shape)
    # x = x.to(diffusion.device)
    # x_noisy, noised = diffusion.noiser(x)
    # print(x_noisy.shape) 
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

    # x = np.transpose(x_noisy, (1, 2, 0))
    # plt.imshow(x)

    # plt.show()     
    # x = x.squeeze(0)
    # x = x.permute(1,2,0)
    # x = x.cpu().detach().numpy()
    # plt.imshow(x)
    # plt.show()
    # plt.imsave('test.png',x)