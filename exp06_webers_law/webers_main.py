import sys
sys.path.insert(1, 'D:/Niranjan_Work/dnns_qualitative/dnn_perception_expts')
import numpy as np
import pickle
from tqdm import tqdm
from _utils.data import load_stim_file, save_stims
from _utils.network import load_model, get_layerwise_activations
import matplotlib.pyplot as plt
import scipy.io as sio
from _utils.stats import nan_corrcoeff

stim_data = load_stim_file('./data/weber_stim.mat') # 20x3x224x224 (distances correspond to i, i+10 for i=0...9)

rel_distances, abs_distances =  sio.loadmat('./data/weber_distances.mat', squeeze_me=True, struct_as_record=True)['distances'].tolist()
#10x1 arrays (distances correspond to pair of images (i, i+10) for i=0...9)


def get_weber_law_correlation(model):
    # extract stim features
    print("Extracting layerwise activations...")
    image_reps = []
    layers = load_model(model)
    for stim_i, stim in enumerate(tqdm(stim_data)):
        img_rep = get_layerwise_activations(stim)
        image_reps.append(img_rep)

    image_reps = np.array(image_reps)
    image_reps = image_reps.reshape((10,2))

    r_absolute = []
    r_relative = []

    print("Computing layerwise correlations...")
    for layer_i, layer in enumerate(tqdm(layers)):
        img1_layer_reps = [] 
        img2_layer_reps = []

        for img_rep in image_reps[:,0]:
            img1_layer_reps.append(img_rep[layer].flatten())

        for img_rep in image_reps[:,1]:
            img2_layer_reps.append(img_rep[layer].flatten())
        
        img1_layer_reps = np.array(img1_layer_reps)
        img2_layer_reps = np.array(img2_layer_reps)
        
        # compute absolute and relative distance correlations
        neural_distance = np.linalg.norm(img1_layer_reps - img2_layer_reps, axis=1)
        
        r_absolute.append(nan_corrcoeff(abs_distances, neural_distance)[0])
        r_relative.append(nan_corrcoeff(rel_distances, neural_distance)[0])

    correlations_diff = np.array(r_relative) - np.array(r_absolute)



    # save correlations
    with open(f'./results/{model}_correlations.pkl', 'wb') as f:
        pickle.dump(correlations_diff, f)
    print("Saved Weber Law Correlations!")

for model in ['vgg16', 'vit_base']:
    get_weber_law_correlation(model)