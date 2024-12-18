import sys
sys.path.insert(1, 'D:/Niranjan_Work/dnns_qualitative/dnn_perception_expts')
import numpy as np
import pickle
from tqdm import tqdm
from _utils.data import load_stim_file
from _utils.network import load_model, get_layerwise_activations
import os

def get_thatcher_scores(model_name, save=False):
    stim_data = load_stim_file('./data/thatcher_faces.mat', model=model_name)
    # Stim_Data arrangement: 20 upright, 20 thatcherized, 20 inverted, 20 thatcherized inverted

    num_stim = 20
    upright_start, upright_thatch_start = 0, 20
    inverted_start, inverted_thatch_start = 40, 60
    layers = load_model(model_name)
    

    
    num_layers = len(layers)
    layerwise_scores = np.zeros((num_stim, num_layers))
    
    for stim_i in tqdm(range(num_stim)):
        upright_i = upright_start + stim_i
        inverted_i = inverted_start + stim_i
        upright_thatch_i = upright_thatch_start + stim_i
        inverted_thatch_i = inverted_thatch_start + stim_i

        # get layer reps for each image
        upright_reps = get_layerwise_activations(stim_data[upright_i])
        inverted_reps = get_layerwise_activations(stim_data[inverted_i])
        upright_thatch_reps = get_layerwise_activations(stim_data[upright_thatch_i])
        inverted_thatch_reps = get_layerwise_activations(stim_data[inverted_thatch_i])
        
        
        # get layerwise scores
        for layer_i in range(num_layers):
            layer_name = layers[layer_i]
            
            upright_dist = np.linalg.norm(upright_reps[layer_name] - upright_thatch_reps[layer_name])
            inverted_dist = np.linalg.norm(inverted_reps[layer_name] - inverted_thatch_reps[layer_name])
            thatcher_index = (upright_dist - inverted_dist) / (upright_dist + inverted_dist)
            layerwise_scores[stim_i, layer_i] = thatcher_index
    
    layerwise_scores_mean = np.nanmean(layerwise_scores, axis=0)
    layerwise_sem = np.nanstd(layerwise_scores, axis=0) / np.sqrt(num_stim)

    # save layerwise scores
    if save:
        with open(f'./results/{model_name}.pkl', 'wb') as f:
            pickle.dump(layerwise_scores, f)
        print("Saved Layerwise Thatcher Effect Scores!")
    return layerwise_scores_mean, layerwise_sem




if __name__ == "__main__":
    accepted_models = ["vgg16", "vit_base"]
    for model in accepted_models:
        get_thatcher_scores(model, save=True)
