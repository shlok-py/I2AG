import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

spectrogram = np.load('/home/shlok/working_data/Ferret/spectrograms/Ferret1.npy')

# Create a figure without axes labels and ticks

plt.imshow(spectrogram, cmap='viridis')  # You can choose a different colormap if needed
plt.axis('off')  # Turn off axes labels and ticks

# Convert the plot to an image
fig = plt.gcf()
fig.canvas.draw()
image_data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
image_data = image_data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

# Close the plot to release resources
plt.close(fig)
image = Image.fromarray(image_data)

image.save('output_spectrogram.png')

# print(spectrogram)