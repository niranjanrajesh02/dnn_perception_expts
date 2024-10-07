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
    
    # Get regression coefficients
    coeff_combined = model.coef_
    
    return coeff_combined

stim_data = load_stim_file('./data/div_norm_stim.mat')
# print(stim_data.shape) # images: (267, 3, 224, 224)
# 49 objects
# singleton => in top, mid or bottom => 3x49 = 147
# pair => in 2 of the 3 positions => 60
# triplet => in 3 of the 3 positions => 60
# total => 267

stim_pos =  sio.loadmat('./data/div_norm_stim_pos.mat', squeeze_me=True, struct_as_record=True)['stim_pos']
# print(stim_pos.shape) # image_positions: (267, 3)
# for each img in stim_data, positions encoded in 3d vector e.g. [1, 2, 999] means top=obj1, mid=obj2, bottom=blank



def get_normalization_slopes(model):
    n_images = 147
    singleton_ind = range(0, n_images)
    pair_ind = range(n_images, n_images + 60)
    pair_i_start = 147
    triplet_ind = range(n_images + 60, n_images + 120)
    triplet_i_start = 207
    var_threshold = 0.1

    layers = load_model(model)
    n_layers = len(layers)
    singleton_representations = []

    print("Computing layerwise activations for singleton images...")
    for stim_i in tqdm(range(n_images)):
        img_rep = get_layerwise_activations(stim_data[stim_i])
        singleton_representations.append(img_rep)

    single_group = np.arange(0, n_images)
    single_group = np.reshape(single_group, (49,3))

    neurons_per_layer = {} #stores n_units per layer
    visually_active_neurons_per_layer = {} #

    # Identify Visually Active Neurons
    print("Identifying visually active neurons...")
    for layer_i in tqdm(range(n_layers)):
        layer_name = layers[layer_i]
        n_units = len(singleton_representations[0][layer_name].flatten())
        neurons_per_layer[layer_name] = n_units
        layerwise_responses = np.zeros((n_images, n_units))

        for stim_i in range(n_images):
            img_rep = singleton_representations[stim_i][layer_name]
            layerwise_responses[stim_i, :] = img_rep.flatten()

        reps_top = layerwise_responses[single_group[:, 0], :]
        reps_mid = layerwise_responses[single_group[:, 1], :]
        reps_bot = layerwise_responses[single_group[:, 2], :]
        
        var_reps_top = np.var(reps_top, axis=0)
        var_reps_mid = np.var(reps_mid, axis=0)
        var_reps_bot = np.var(reps_bot, axis=0)
        
        van_indices = np.where((var_reps_top > var_threshold) & (var_reps_mid > var_threshold) & (var_reps_bot > var_threshold))[0]
        visually_active_neurons_per_layer[layer_name] = van_indices

    active_layers = []
    active_layer_inds = []
    for layer_i, layer in enumerate(layers):
        if len(visually_active_neurons_per_layer[layer]) > 1:
            active_layers.append(layer)
            active_layer_inds.append(layer_i)

    model_layers = [active_layers, active_layer_inds]

    # save
    with open(f'./results/{model}_layers.pkl', 'wb') as f:
        pickle.dump(model_layers, f)

    print(f"Found {len(active_layers)} layers in model {model}!")

    pair_representations = []
    triplet_representations = []



    # Extract features for pairs, triplets
    print("Computing layerwise activations for pair images...")
    for stim_i, stim in enumerate(tqdm(stim_data[pair_ind])):
        img_rep = get_layerwise_activations(stim)
        pair_representations.append(img_rep)


    print("Computing normalization slopes for pair images...")
    # computing normalization slopes
    pair_stim_slopes = np.ones((60,len(active_layers)))
    for stim_i in tqdm(range(60)):
        individual_representations = []
        combined_representations = pair_representations[stim_i] 
        for pos_i in range(3):
            # print(stim_pos[pair_i_start + stim_i, pos_i])
            if stim_pos[pair_i_start + stim_i, pos_i] != 999:
                    singleton_stim_ind = np.where(stim_pos[singleton_ind][:, pos_i] == stim_pos[pair_ind][stim_i, pos_i])[0][0]
                    individual_representations.append(singleton_representations[singleton_stim_ind])

        for layer_i, layer in enumerate(active_layers):
            van_indices = visually_active_neurons_per_layer[layer]
            f_single = individual_representations[0][layer].flatten()[van_indices] + individual_representations[1][layer].flatten()[van_indices]
            f_pair = combined_representations[layer].flatten()[van_indices]
            coeff = find_normalization_combined(f_pair.numpy(), f_single.numpy())
            pair_stim_slopes[stim_i, layer_i] = coeff[0]

    layerwise_pair_slopes = np.mean(pair_stim_slopes, axis=0)

    print("Computing layerwise activations for triplet images...")
    for stim_i, stim in enumerate(tqdm(stim_data[triplet_ind])):
        img_rep = get_layerwise_activations(stim)
        triplet_representations.append(img_rep)

    print("Computing normalization slopes for triplet images...")
    # computing normalization slopes
    triplet_stim_slopes = np.ones((60,len(active_layers)))
    for stim_i in tqdm(range(60)):
        individual_representations = []
        combined_representations = triplet_representations[stim_i] 
        for pos_i in range(3):
            # print(stim_pos[pair_i_start + stim_i, pos_i])
            if stim_pos[triplet_i_start + stim_i, pos_i] != 999:
                    singleton_stim_ind = np.where(stim_pos[singleton_ind][:, pos_i] == stim_pos[triplet_ind][stim_i, pos_i])[0][0]
                    individual_representations.append(singleton_representations[singleton_stim_ind])

        for layer_i, layer in enumerate(active_layers):
            van_indices = visually_active_neurons_per_layer[layer]
            f_single = individual_representations[0][layer].flatten()[van_indices] + individual_representations[1][layer].flatten()[van_indices] + individual_representations[2][layer].flatten()[van_indices]
            f_combined = combined_representations[layer].flatten()[van_indices]
            coeff = find_normalization_combined(f_combined.numpy(), f_single.numpy())
            triplet_stim_slopes[stim_i, layer_i] = coeff[0]

    layerwise_triplet_slopes = np.mean(triplet_stim_slopes, axis=0)

    layerwise_slopes = [layerwise_pair_slopes, layerwise_triplet_slopes]
    # pickle save
    with open(f'./results/{model}.pkl', 'wb') as f:
        pickle.dump(layerwise_slopes, f)

models = ['vgg16', 'vit_base']

for model in models:
    get_normalization_slopes(model)