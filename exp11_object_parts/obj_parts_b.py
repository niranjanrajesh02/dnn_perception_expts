import sys
sys.path.insert(1, 'D:/Niranjan_Work/dnns_qualitative/dnn_perception_expts')
import numpy as np
import pickle
from tqdm import tqdm
from _utils.data import load_stim_file, save_stims
from _utils.network import load_model, get_layerwise_activations
from scipy.io import loadmat
from sklearn.linear_model import LinearRegression
from _utils.stats import nan_corrcoeff


def get_part_correlations(model, save=False):
    stim_array = load_stim_file('./data/obj_parts_b_stim.mat', stim_var='images', model=model) 
    # print(stim_array.shape) # 98x3x224x224
    # 2 sets of 49 images:
    # set 1: (unnatural part variation) the left and right part varies;  every 7 images, the left part varies. for each left part, there are 7 right part variations
    # set 2: (natural part variation) the top and bottom part varies;  every 7 images, the top part varies. for each top part, there are 7 bottom part variations

    matfile = loadmat('./data/data_matrices.mat')['data_matrices']

    X_natural = matfile[0][0][0] # 492x43 design matrix for natural pairs (cols encode the different parts of both stims)
    X_unnatural = matfile[0][0][1] # 492x43 design matrix for unnatural pairs
    img_pair_inds = matfile[0][0][2] # 984x2 matrix where all possible pairs of stimuli indices are stored
    common_stim_pair_inds = matfile[0][0][3] #corresponding indices in above matrix where common stim pairs are stored (pairs that contain two of the 'common stim'- in both natural and unnatural)
    # used as test set


    # subtract ind arrays by 1 for python indexing
    img_pair_inds = img_pair_inds - 1
    common_stim_pair_inds = common_stim_pair_inds - 1

    layers = load_model(model)

    img_representations = []


    print("Extracting layerwise activations...")
    for stim_i, stim in enumerate(tqdm(stim_array)):
        img_rep = get_layerwise_activations(stim)
        img_representations.append(img_rep)

    img_representations = np.array(img_representations)

    pair_distances = np.zeros((len(img_pair_inds), len(layers)))

    for image_pair_i in tqdm(range(len(img_pair_inds))):
        img_pair = img_pair_inds[image_pair_i]
        for layer_i, layer in enumerate(layers):
            f1 = img_representations[img_pair[0]][layer].flatten()
            f2 = img_representations[img_pair[1]][layer].flatten()
            pair_distances[image_pair_i, layer_i] = np.linalg.norm(f1 - f2)


    # model fitting
    r_unnatural = np.zeros((len(layers)))
    r_natural = np.zeros((len(layers)))

    q_unnatural = np.arange(0, 492) # pairs of unnatural stim
    q_natural = np.arange(0, 492) + 492 # pairs of natural stim


    # fit two models for unnatural and natural pairs and test on common set
    for layer_i, layer in enumerate(layers):
        unnatural_distances = pair_distances[q_unnatural, layer_i]
        u_model = LinearRegression().fit(X_unnatural, unnatural_distances)
        predicted_u_distances = u_model.predict(X_unnatural)

        natural_distances = pair_distances[q_natural, layer_i]
        n_model = LinearRegression().fit(X_natural, natural_distances)
        predicted_n_distances = n_model.predict(X_natural)

        r_u = nan_corrcoeff(unnatural_distances[common_stim_pair_inds], predicted_u_distances[common_stim_pair_inds])
        r_n = nan_corrcoeff(natural_distances[common_stim_pair_inds], predicted_n_distances[common_stim_pair_inds])
        
        
        r_unnatural[layer_i] = r_u[0]
        r_natural[layer_i] = r_n[0]

    natural_advantage = r_natural - r_unnatural

    if save:
        with open(f'./results/{model}_natural_advantage.pkl', 'wb') as f:
            pickle.dump(natural_advantage, f)
    return np.array(natural_advantage)

if __name__ == "__main__":
    for model in ['vgg16', 'vit_base']:
        get_part_correlations(model)