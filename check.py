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

# print((torch.ones(8) * 1000).long())
# print(torch.randn((8,3,5,5)))
# x = torch.randn((8,3,128,128))
# print(torch.zeros_like(x))
# x = (x.clamp(-1,1) + 1) /2
# x = x * 255
# image_array = x[:1,:,:,:].reshape((3, 128, 128))

# Transpose the array to (5, 5, 3) to match the standard image format
# image_array = np.transpose(image_array, (1, 2, 0))
# plt.imshow(image_array)

# plt.show()
# print(x[:1, :, :, :])

beta = torch.linspace(0.001,0.02, 1000)
# print(beta)
alpha = 1 - beta
# print(alpha)
alpha_hat = torch.cumprod(alpha, dim = 0)
print(torch.sqrt(alpha_hat[999])[:, None, None, None])