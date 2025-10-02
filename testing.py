#%%
import torch
import matplotlib.pyplot as plt

data = torch.load('dataset_10episodes_random_spaceinvaders.pt', weights_only=False)

plt.imshow(data['frames'][1000])
plt.show()
