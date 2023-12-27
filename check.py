import os
import numpy as np
import pickle
import torch
import matplotlib.pyplot as plt
# audio_files_dir = "/home/shlok/working_data/Ferret/audio"
# # print(os.walk(audio_files_dir))
# root = os.walk(audio_files_dir)
# # print(root)
# for root,_,files in os.walk(audio_files_dir):
#     print("root = ",root)
#     for file in files:
#         print(os.path.join(root,file))
# dir = "/home/shlok/working_data/Leopard/spectrograms/"
# for root,_,files in os.walk(dir):
#     for file in files:
#         c = os.path.join(root,file)
#         a = np.load(c)
#         print(a.shape)

# print(len(torch.randint(low=1, high = 1000, size = (1000,))))
# print(len(1/(10000**(torch.arange(0, 256, 2)/ 256))))
# def pos_enc(t, channels):
#     inv_freq = 1.0 / (
#         10000
#         ** (torch.arange(0, channels, 2).float() / channels)
#     )
#     pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
#     print('pos encoding_a',pos_enc_a.shape)
#     pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
#     print('pos encoding_b',pos_enc_b.shape)
#     pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
#     return pos_enc


# t = torch.randint(low=1, high = 1000, size = (1000,))
# # print("1:-------------\n",len(t))
# t = t.unsqueeze(-1)
# # print("2:-------------\n",len(t))
# t = pos_enc(t, 256)
# print("3------------\n", t.shape)
# # print(t.repeat(1, 256//2).shape)
# print(5//2)
# import torch

# Assuming you have the time_space and latent_space tensors
# time_space = torch.randn(4, 256)  # Example tensor with shape [4, 256]
# latent_space = torch.randn(4, 128)  # Example tensor with shape [4, 128]

# result = time_space + latent_space.unsqueeze(1)  # Unsqueeze to add a new dimension along axis 1

# Print the result
# print(result.shape)  # Shape of the result tensor
# print(result)
x = torch.tensor(np.random.rand(2,128,128,128))
x = x.view(-1, 128*16*16, 2 * 2).swapaxes(1, 2)
print(x.shape)
x = x.swapaxes(2,1).view(-1, 128, 2,2)
print(x.shape)