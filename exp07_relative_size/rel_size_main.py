import sys
sys.path.insert(1, 'D:/Niranjan_Work/dnns_qualitative/dnn_perception_expts')
import numpy as np
import pickle
from tqdm import tqdm
from _utils.data import load_stim_file, save_stims
from _utils.network import load_model, get_layerwise_activations
import matplotlib.pyplot as plt
from _utils.stats import nan_corrcoeff, normalize

stim_data = load_stim_file('./data/relSize.mat')
# 96x3x224x224
n_tetrad_groups = 24
# each group has 4 imgs with 2 parts each of varying sizes
# img1 part1: size 1, part2: size 1 (CONGRUENT)
# img2 part1: size 1, part2: size 2 (INCONGRUENT)
# img3 part1: size 2, part2: size 1 (INCONGRUENT)
# img4 part1: size 2, part2: size 2 (CONGRUENT)




# extract stim features
model = 'vgg16'
print("Extracting layerwise activations...")
image_reps = []
layers = load_model(model)
for stim_i, stim in enumerate(tqdm(stim_data)):
    img_rep = get_layerwise_activations(stim)
    image_reps.append(img_rep)

image_reps = np.array(image_reps)

for layer_i, layer in enumerate(tqdm(layers)):
    layerwise_reps = []
    for img_rep in image_reps:
        layerwise_reps.append(img_rep[layer].flatten())
    layerwise_reps = np.array(layerwise_reps)
    normalized_layerwise_reps = normalize(layerwise_reps)
    
    van_indices = np.where(np.sum(normalized_layerwise_reps, axis=0) > 0)[0]
    normalized_layerwise_reps = normalized_layerwise_reps[:, van_indices]

    for group_i in range(n_tetrad_groups):
        img_inds = group_i * 4 + np.arange(4)
        img1_resp = normalized_layerwise_reps[img_inds[0]]
        img2_resp = normalized_layerwise_reps[img_inds[1]]
        img3_resp = normalized_layerwise_reps[img_inds[2]]
        img4_resp = normalized_layerwise_reps[img_inds[3]]

        temp_sum = np.array([img1_resp, img2_resp, img3_resp, img4_resp])
        temp_sum = np.sum(temp_sum, axis=0)
        
        n_active_tetrad_units = len(np.where(np.sum(temp_sum) > 0)[0])
        # nan the inactive tetrad units
        inactive_tetrad_units = np.where(np.sum(temp_sum) <= 0)[0]
        img1_resp[inactive_tetrad_units] = np.nan
        img2_resp[inactive_tetrad_units] = np.nan
        img3_resp[inactive_tetrad_units] = np.nan
        img4_resp[inactive_tetrad_units] = np.nan

        mr1 = (img1_resp + img2_resp)/2
        mr2 = (img3_resp + img4_resp)/2
        mc1 = (img1_resp + img3_resp)/2
        mc2 = (img2_resp + img4_resp)/2

        T = np.array([img1_resp, img2_resp, img3_resp, img4_resp])
        residual_error = T + np.mean(T, axis=0) - np.array([mr1, mr2, mc1, mc2]) - np.array([mc1, mc2, mr1, mr2])
        
        break
    break
