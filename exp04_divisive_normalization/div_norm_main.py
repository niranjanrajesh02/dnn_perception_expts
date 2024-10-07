import sys
sys.path.insert(1, 'D:/Niranjan_Work/dnns_qualitative/expts_py')
import numpy as np
import pickle
import scipy.io as sio
from tqdm import tqdm
from _utils.data import load_multiple_stim_files, load_stim_file
from _utils.network import load_model, get_layerwise_activations
import matplotlib.pyplot as plt

stim_data = load_stim_file('./data/div_norm_stim.mat')
print(stim_data.shape) # images: (267, 3, 224, 224)
# 49 objects
# singleton => in top, mid or bottom => 3x49 = 147
# pair => in 2 of the 3 positions => 60
# triplet => in 3 of the 3 positions => 60
# total => 267

stim_pos =  sio.loadmat('./data/div_norm_stim_pos.mat', squeeze_me=True, struct_as_record=True)['stim_pos']
print(stim_pos.shape) # image_positions: (267, 3)
# for each img in stim_data, positions encoded in 3d vector e.g. [1, 2, 999] means top=obj1, mid=obj2, bottom=blank

# save images in stim_data in'./results' folder
for i in tqdm(range(267)):
    img = stim_data[i, :, :, :]
    img = np.transpose(img, (1, 2, 0))
    plt.imsave('./images/' + str(i) + '.png', img, cmap='gray')
