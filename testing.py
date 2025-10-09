#%%
import torch
import matplotlib.pyplot as plt
import pickle
import numpy as np
from skimage.transform import downscale_local_mean
from actions_to_tokens import act_to_tok

with open('panorama_dataset.pkl', 'rb') as f:
    data = pickle.load(f)

plt.imshow(np.round(downscale_local_mean(data['sessions'][2]['frames'][0], (2, 2, 1))).astype(np.uint8))
plt.show()
print(np.round(downscale_local_mean(data['sessions'][2]['frames'][0], (2, 2, 1))).astype(np.uint8).shape)
print(set([tuple(x) for x in data['sessions'][2]['actions']]))
print(np.unique([act_to_tok[tuple(x)] for x in data['sessions'][2]['actions']], return_counts=True))

# %%
print(len(data['actions']))
# %%
import matplotlib.pyplot as plt
import numpy as np

data = np.load('tmp_lpip.npy', mmap_mode='c')

plt.plot(data)
plt.ylabel('LPIPS (lower is better)')
plt.xlabel('Auto-regressive step')
plt.legend(['human agent'], loc="lower right")
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
plt.legend(['human agent'], loc="lower right")
plt.show()
# %%
import patoolib
patoolib.extract_archive("test.rar", outdir=".")