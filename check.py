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
dir = "/home/shlok/working_data/Leopard/spectrograms/"
for root,_,files in os.walk(dir):
    for file in files:
        c = os.path.join(root,file)
        a = np.load(c)
        print(a.shape)