import sys
sys.path.insert(1, 'D:/Niranjan_Work/dnns_qualitative/dnn_perception_expts')
import numpy as np
import pickle
from tqdm import tqdm
from _utils.data import load_stim_file, save_stims
from _utils.network import load_model, get_layerwise_activations
import matplotlib.pyplot as plt
from _utils.stats import nan_corrcoeff, normalize
import warnings 
warnings.filterwarnings("ignore")





# extract stim features

def get_relative_size_index(model, save=False):
    stim_data = load_stim_file('./data/relSize.mat',model=model)
    # 96x3x224x224
    n_tetrad_groups = 24
    # each group has 4 imgs with 2 parts each of varying sizes
    # img1 part1: size 1, part2: size 1 
    # img2 part1: size 1, part2: size 2 
    # img3 part1: size 2, part2: size 1 
    # img4 part1: size 2, part2: size 2 

    # img1,img4 => congruent pair; img2,img3 => incongruent pair

    print("Extracting layerwise activations...")
    image_reps = []
    layers = load_model(model)
    for stim_i, stim in enumerate(tqdm(stim_data)):
        img_rep = get_layerwise_activations(stim)
        image_reps.append(img_rep)

    image_reps = np.array(image_reps)

    relative_size_index = []
    relative_size_index_sem = []
    n_tetrads = np.zeros((len(layers), n_tetrad_groups))
    print("Computing layerwise relative size index...")
    for layer_i, layer in enumerate(tqdm(layers)):
        layerwise_reps = []
        for img_rep in image_reps:
            layerwise_reps.append(img_rep[layer].flatten())
        layerwise_reps = np.array(layerwise_reps)
        normalized_layerwise_reps = normalize(layerwise_reps)
        
        van_indices = np.where(np.sum(normalized_layerwise_reps, axis=0) > 0)[0]
        normalized_layerwise_reps = normalized_layerwise_reps[:, van_indices]
        layer_re = []
        layer_rsi = []


        for group_i in range(n_tetrad_groups):
            img_inds = group_i * 4 + np.arange(4)
            img1_resp = normalized_layerwise_reps[img_inds[0]]
            img2_resp = normalized_layerwise_reps[img_inds[1]]
            img3_resp = normalized_layerwise_reps[img_inds[2]]
            img4_resp = normalized_layerwise_reps[img_inds[3]]
        
            # identify active and inactive units for this tetrad
            temp_sum = np.array([img1_resp, img2_resp, img3_resp, img4_resp])
            temp_sum = np.sum(temp_sum, axis=0)

            active_tetrad_units = np.where(temp_sum > 0)[0]
            inactive_tetrad_units = np.where(temp_sum <= 0)[0]
            n_tetrads[layer_i, group_i] = len(active_tetrad_units)


            # nan the inactive tetrad units
            img1_resp[inactive_tetrad_units] = np.nan
            img2_resp[inactive_tetrad_units] = np.nan
            img3_resp[inactive_tetrad_units] = np.nan
            img4_resp[inactive_tetrad_units] = np.nan

            mr1 = (img1_resp + img2_resp)/2
            mr2 = (img3_resp + img4_resp)/2
            mc1 = (img1_resp + img3_resp)/2
            mc2 = (img2_resp + img4_resp)/2

            T = np.array([img1_resp, img2_resp, img3_resp, img4_resp])
            residual_error = T + np.mean(T, axis=0) - np.array([mr1, mr1, mr2, mr2]) - np.array([mc1, mc2, mc1, mc2])
            residual_error = np.abs(np.sum(residual_error, axis=0))

            layer_re.append(residual_error)
            
            d14 = np.abs(img1_resp - img4_resp)
            d23 = np.abs(img2_resp - img3_resp)

            rel_size_index = (d14 - d23) / (d14 + d23)
            layer_rsi.append(rel_size_index)
            
        # select top 7%  tetrad units with highest residual error
        layer_re = np.array(layer_re).T #n_neuronsx24
        layer_rsi = np.array(layer_rsi).T #n_neuronsx24

        layer_re = layer_re.flatten()
        layer_rsi = layer_rsi.flatten()

        layer_re[np.isnan(layer_re)] = -9999999
        # sorted vals and inds 
        # sorted_layer_re = np.sort(layer_re)[::-1]
        sorted_layer_inds = np.argsort(layer_re)[::-1]

        total_active_tetrads = np.sum(n_tetrads[layer_i])

        num_top_tetrads = int(np.floor(0.07*total_active_tetrads))

        selected_layer_rsi = layer_rsi[sorted_layer_inds[0:num_top_tetrads]]
        
        layer_rsi_score = np.nanmean(selected_layer_rsi)
        relative_size_index.append(layer_rsi_score)

        layer_rsi_sem = np.nanstd(selected_layer_rsi) / np.sqrt(num_top_tetrads)
        relative_size_index_sem.append(layer_rsi_sem)

        
    if save:   
        with open(f'./results/{model}_rsi.pkl', 'wb') as f:
            pickle.dump(relative_size_index, f)
        print("Saved relative size index results for model: " + model)
    return np.array(relative_size_index), np.array(relative_size_index_sem)

if __name__ == '__main__':
    for model in ['vgg16', 'vit_base']:
        get_relative_size_index(model)