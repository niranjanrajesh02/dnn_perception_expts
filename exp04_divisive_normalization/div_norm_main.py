import sys
sys.path.insert(1, 'D:/Niranjan_Work/dnns_qualitative/dnn_perception_expts')
import numpy as np
import pickle
import scipy.io as sio
from tqdm import tqdm
from _utils.data import load_multiple_stim_files, load_stim_file
from _utils.network import load_model, get_layerwise_activations
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import gc
import torch

def find_normalization_combined(Fcombined, Fsum):
    fc_max = np.max(Fcombined)
    fsum_max = np.max(Fsum)
    C = max(fc_max, fsum_max)
    
    # Prepare the data for regression
    X = np.column_stack([Fsum.flatten()/C, np.ones(len(Fsum.flatten()))])
    y = Fcombined.flatten() /C
    
    # Perform linear regression
    model = LinearRegression(fit_intercept=False)  # No intercept because it's included in X
    model.fit(X, y)
    
    
    coeff_combined = model.coef_
    
    return coeff_combined





def get_div_norm_scores(model, save=False):
    stim_data = load_stim_file('./data/div_norm_stim.mat', model=model)
    # print(stim_data.shape) # images: (267, 3, 224, 224)
    # 49 objects
    # singleton => in top, mid or bottom => 3x49 = 147
    # pair => in 2 of the 3 positions => 60
    # triplet => in 3 of the 3 positions => 60
    # total => 267

    stim_pos =  sio.loadmat('./data/div_norm_stim_pos.mat', squeeze_me=True, struct_as_record=True)['stim_pos']
    # print(stim_pos.shape) # image_positions: (267, 3)
    # for each img in stim_data, positions encoded in 3d vector e.g. [1, 2, 999] means top=obj1, mid=obj2, bottom=blank
    n_images = 147
    singleton_ind = range(0, n_images)
    pair_ind = range(n_images, n_images + 60)
    pair_i_start = 147
    triplet_ind = range(n_images + 60, n_images + 120)
    triplet_i_start = 207
    var_threshold = 0.1

    layers = load_model(model)
    n_layers = len(layers)


    single_group = np.arange(0, n_images)
    single_group = np.reshape(single_group, (49,3))

    neurons_per_layer = {} #stores n_units per layer
    visually_active_neurons_per_layer = {} #

    # Identify Visually Active Neurons
    print("Identifying visually active neurons...")
    # visually active neurons: neurons that are active in all 3 positions in each layer

    reps_top = []
    top_vars = []

    print("Top Variances...")
    for i in tqdm(range(49)):
            top_img = get_layerwise_activations(stim_data[single_group[i, 0]])
            reps_top.append(top_img)
    for layer_i in tqdm(range(n_layers)):
        layer_name = layers[layer_i]
        n_units = len(reps_top[0][layer_name].flatten())
        neurons_per_layer[layer_name] = n_units
        # find var
        layer_reps = []
        for i in range(49):
            rep = reps_top[i][layer_name].flatten()
            layer_reps.append(rep)
        top_var = np.var(layer_reps, axis=0)
        top_vars.append(top_var)
    del reps_top
    gc.collect()
    
    reps_mid = []
    mid_vars = []
    print("Mid Variances...")
    for i in tqdm(range(49)):
            mid_img = get_layerwise_activations(stim_data[single_group[i, 1]])
            reps_mid.append(mid_img)
    for layer_i in tqdm(range(n_layers)):
        layer_name = layers[layer_i]
        n_units = len(reps_mid[0][layer_name].flatten())
        neurons_per_layer[layer_name] = n_units
        # find var
        layer_reps = []
        for i in range(49):
            rep = reps_mid[i][layer_name].flatten()
            layer_reps.append(rep)
        mid_var = np.var(layer_reps, axis=0)
        mid_vars.append(mid_var)
    del reps_mid
    gc.collect()

    reps_bot = []
    bot_vars = []
    print("Bottom Variances...")
    for i in tqdm(range(49)):
            bot_img = get_layerwise_activations(stim_data[single_group[i, 2]])
            reps_bot.append(bot_img)
    
    # bottom var + vans
    for layer_i in tqdm(range(n_layers)):
        layer_name = layers[layer_i]
        n_units = len(reps_bot[0][layer_name].flatten())
        neurons_per_layer[layer_name] = n_units
        # find var
        layer_reps = []
        for i in range(49):
            rep = reps_bot[i][layer_name].flatten()
            layer_reps.append(rep)
        bot_var = np.var(layer_reps, axis=0)
        bot_vars.append(bot_var)

        # find visually active neurons
        visually_active_neurons_per_layer[layer_name] = np.where((top_vars[layer_i] > var_threshold) & (mid_vars[layer_i] > var_threshold) & (bot_vars[layer_i] > var_threshold))[0]
        


    active_layers = []
    active_layer_inds = []
    for layer_i, layer in enumerate(layers):
        if len(visually_active_neurons_per_layer[layer]) > 10:
            active_layers.append(layer)
            active_layer_inds.append(layer_i)

    model_layers = [active_layers, active_layer_inds]
 
    # save
    if save:
        with open(f'./results/{model}_layers.pkl', 'wb') as f:
            pickle.dump(model_layers, f)

        print(f"Found {len(active_layers)} active layers in model {model}!")

 

    print("Computing normalization slopes for pair images...")
    # computing normalization slopes
    pair_stim_slopes = np.ones((60,len(active_layers)))
    for stim_i in tqdm(range(60)):
        individual_representations = []
        combined_representations = get_layerwise_activations(stim_data[pair_ind][stim_i])
       
        for pos_i in range(3):
            # print(stim_pos[pair_i_start + stim_i, pos_i])
            if stim_pos[pair_i_start + stim_i, pos_i] != 999:
                    singleton_stim_ind = np.where(stim_pos[singleton_ind][:, pos_i] == stim_pos[pair_ind][stim_i, pos_i])[0][0]
                    individual_representations.append(get_layerwise_activations(stim_data[singleton_stim_ind]))

        for layer_i, layer in enumerate(active_layers):
            van_indices = visually_active_neurons_per_layer[layer]
            f_single = individual_representations[0][layer].flatten()[van_indices] + individual_representations[1][layer].flatten()[van_indices]
            f_pair = combined_representations[layer].flatten()[van_indices]
            coeff = find_normalization_combined(f_pair.numpy(), f_single.numpy())
            
            pair_stim_slopes[stim_i, layer_i] = coeff[0]

    layerwise_pair_slopes = np.mean(pair_stim_slopes, axis=0)
    pair_slopes = np.empty(len(layers))
    pair_slopes[:] = np.nan
    pair_slopes[active_layer_inds] = layerwise_pair_slopes
    pair_slopes_sem = np.std(pair_stim_slopes, axis=0) / np.sqrt(60)

    

    print("Computing normalization slopes for triplet images...")
    # computing normalization slopes
    triplet_stim_slopes = np.ones((60,len(active_layers)))
    for stim_i in tqdm(range(60)):
        individual_representations = []
        combined_representations = get_layerwise_activations(stim_data[triplet_ind][stim_i])
        
        for pos_i in range(3):
            # print(stim_pos[pair_i_start + stim_i, pos_i])
            if stim_pos[triplet_i_start + stim_i, pos_i] != 999:
                    singleton_stim_ind = np.where(stim_pos[singleton_ind][:, pos_i] == stim_pos[triplet_ind][stim_i, pos_i])[0][0]
                    individual_representations.append(get_layerwise_activations(stim_data[singleton_stim_ind]))

        for layer_i, layer in enumerate(active_layers):
            van_indices = visually_active_neurons_per_layer[layer]
            f_single = individual_representations[0][layer].flatten()[van_indices] + individual_representations[1][layer].flatten()[van_indices] + individual_representations[2][layer].flatten()[van_indices]
            f_combined = combined_representations[layer].flatten()[van_indices]
            coeff = find_normalization_combined(f_combined.numpy(), f_single.numpy())
            triplet_stim_slopes[stim_i, layer_i] = coeff[0]

    layerwise_triplet_slopes = np.mean(triplet_stim_slopes, axis=0)
    triplet_slopes = np.empty(len(layers))
    triplet_slopes[:] = np.nan
    triplet_slopes[active_layer_inds] = layerwise_triplet_slopes
    triplet_slopes_sem = np.std(triplet_stim_slopes, axis=0) / np.sqrt(60)

    layerwise_slopes = [pair_slopes, triplet_slopes]
    layerwise_slopes_sem = [pair_slopes_sem, triplet_slopes_sem]
    
    
    if save:
        with open(f'./results/{model}.pkl', 'wb') as f:
            pickle.dump(layerwise_slopes, f)
    
    return np.array(layerwise_slopes), np.array(layerwise_slopes_sem)

if __name__ == "__main__":
    models = ['face_vit']

    for model in models:
        get_div_norm_scores(model, save=True)