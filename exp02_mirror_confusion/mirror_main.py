import sys
sys.path.insert(1, 'D:/Niranjan_Work/dnns_qualitative/dnn_perception_expts')
import numpy as np
import os
import pickle
from tqdm import tqdm
from _utils.data import load_stim_file
from _utils.network import load_model, get_layerwise_activations


def get_mirror_scores(model_name, save=False):
    stim_data = load_stim_file('./data/mirror_stim.mat')
    # Stim_Data arrangement: 100 stim, 100 mirror about y-axis, 100 mirror about x-axis

    num_stim = 100
    original_start = 0
    y_flipped_start = 0 + num_stim
    x_flipped_start = 0 + 2 * num_stim

    # load model
    # get num layers from model
    layers = load_model(model_name)
    num_layers = len(layers)

    # init layerwise mirror scores array
    layerwise_scores = np.zeros((num_stim, num_layers))
    for stim_i in tqdm(range(num_stim)):
        original_i = original_start + stim_i
        y_flipped_i = y_flipped_start + stim_i
        x_flipped_i = x_flipped_start + stim_i

        # get layer reps for each image
        original_reps = get_layerwise_activations(stim_data[original_i])
        y_flipped_reps = get_layerwise_activations(stim_data[y_flipped_i])
        x_flipped_reps = get_layerwise_activations(stim_data[x_flipped_i])

        # get layerwise scores
        for layer_i in range(num_layers):
            layer_name = layers[layer_i]
            dist_y = np.linalg.norm(y_flipped_reps[layer_name] - original_reps[layer_name])
            dist_x = np.linalg.norm(x_flipped_reps[layer_name] - original_reps[layer_name])
            mirror_index = (dist_x - dist_y) / (dist_x + dist_y)
            layerwise_scores[stim_i, layer_i] = mirror_index

    layerwise_scores_mean = np.nanmean(layerwise_scores, axis=0)
    layerwise_sem = np.nanstd(layerwise_scores, axis=0) / np.sqrt(num_stim)

    # save layerwise scores
    if save:
        with open(f'./results/{model_name}.pkl', 'wb') as f:
            pickle.dump(layerwise_scores, f)
        print("Saved Layerwise Mirror Confusion Scores!")

    return layerwise_scores_mean, layerwise_sem



if __name__ == "__main__":
    accepted_models = ["vgg16", "vit_base"]
    for model in accepted_models:
        get_mirror_scores(model, save=True)
