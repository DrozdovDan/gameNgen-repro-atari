#%%
import torch
import matplotlib.pyplot as plt
import pickle
import numpy as np

with open('panorama_dataset.pkl', 'rb') as f:
    data = pickle.load(f)

plt.imshow(data['sessions'][0]['frames'][90])
plt.show()
print(data['sessions'][0]['frames'][0].shape)
print(set(list(map(x: tuple(x), data['sessions'][0]['actions']))))

# %%
print(len(data['actions']))
# %%
import matplotlib.pyplot as plt
import numpy as np

data = np.load('tmp.npy', mmap_mode='c')
data_random = np.load('tmp_random.npy', mmap_mode='c')

plt.plot(data)
plt.plot(data_random)
plt.ylabel('LPIPS (lower is better)')
plt.xlabel('Auto-regressive step')
plt.legend(['rl agent', 'random agent'], loc="lower right")
plt.show()


# %%
import matplotlib.pyplot as plt
import numpy as np
data = np.load('tmp_psnr.npy', mmap_mode='c')
#data_random = np.load('tmp_random.npy', mmap_mode='c')

plt.plot(data)
#plt.plot(data_random)
plt.ylabel('PSNR (higher is better)')
plt.xlabel('Auto-regressive step')
plt.legend(['rl agent'], loc="lower right")
plt.show()
# %%
