import sys
sys.path.insert(1, 'D:/Niranjan_Work/dnns_qualitative/dnn_perception_expts')
import numpy as np
import pickle
from tqdm import tqdm
from _utils.data import load_stim_file, save_stims
from _utils.network import load_model, get_layerwise_activations
import matplotlib.pyplot as plt
from _utils.stats import nan_corrcoeff, normalize

stim_array = load_stim_file('./data/3d.mat')
print(stim_array.shape)
# 12 stim (2 sets of 6 stim)
# in one set, img0 and img1 are 2d pair, img2 img 3 are 3d pair, img4 and img5 are 2d pair 
# in one set, pairs only differ by one shape component (like a "Y" shape)


def get_3d_scores(model):
    # extract stim reps
    layers = load_model(model)
    img_reps = []
    print("Extracting layerwise activations...")
    for stim_i, stim in enumerate(tqdm(stim_array)):
        img_rep = get_layerwise_activations(stim)
        img_reps.append(img_rep)
    img_reps = np.array(img_reps)

    layerwise_scores = np.zeros((len(layers), 2))
    for layer_i, layer in enumerate(tqdm(layers)):
        layerwise_reps = []
        
        for img_rep in img_reps:
            layerwise_reps.append(img_rep[layer].flatten())

        # set 1
        pair1_dist = np.linalg.norm(layerwise_reps[0] - layerwise_reps[1])
        pair2_dist = np.linalg.norm(layerwise_reps[2] - layerwise_reps[3]) #3d pair
        pair3_dist = np.linalg.norm(layerwise_reps[4] - layerwise_reps[5])

        condition_1_index = (pair2_dist - pair1_dist) / (pair2_dist + pair1_dist)
        condition_2_index = (pair2_dist - pair3_dist) / (pair2_dist + pair3_dist)

        # set 2
        set2_pair1_dist = np.linalg.norm(layerwise_reps[6] - layerwise_reps[7])
        set2_pair2_dist = np.linalg.norm(layerwise_reps[8] - layerwise_reps[9]) #3d pair
        set2_pair3_dist = np.linalg.norm(layerwise_reps[10] - layerwise_reps[11])
            
        set2_condition_1_index = (set2_pair2_dist - set2_pair1_dist) / (set2_pair2_dist + set2_pair1_dist)
        set2_condition_2_index = (set2_pair2_dist - set2_pair3_dist) / (set2_pair2_dist + set2_pair3_dist)
        
        avg_condition_1_index = (condition_1_index + set2_condition_1_index) / 2
        avg_condition_2_index = (condition_2_index + set2_condition_2_index) / 2
        
        layerwise_scores[layer_i, 0] = avg_condition_1_index
        layerwise_scores[layer_i, 1] = avg_condition_2_index

    with open(f'./results/{model}.pkl', 'wb') as f:
        pickle.dump(layerwise_scores, f)
    print("Saved Layerwise Scores!")


for model in ['vgg16', 'vit_base']:
    get_3d_scores(model)