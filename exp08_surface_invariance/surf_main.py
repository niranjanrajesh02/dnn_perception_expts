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


stim_data = load_stim_file('./data/surf_stim.mat')
# 78x3x224x224
# (6 objects on 5 different surfaces)
# for each object, 13 different variants 0 is base obj on base surface, next 4 triplets are of the following with different alt_surfs (base_obj on alt_surf, alt_obj on base_surf, alt_obj on alt_surf)

n_object_sets = 6
n_stim_per_set = 13
n_triplets = 4 #tetrad = (base_obj on base_surf (0), base_obj on alt_surf, alt_obj on base_surf, alt_obj on alt_surf)

def get_surface_invariance_index(model):
    print("Extracting layerwise activations...")
    image_reps = []
    layers = load_model(model)
    for stim_i, stim in enumerate(tqdm(stim_data)):
        img_rep = get_layerwise_activations(stim)
        image_reps.append(img_rep)

    image_reps = np.array(image_reps)

    surface_invariance_index = []
    n_tetrads = np.zeros((len(layers), n_object_sets*n_triplets))
    print("Computing layerwise surface invariance index...")
    for layer_i, layer in enumerate(tqdm(layers)):
        layerwise_reps = []
        for img_rep in image_reps:
            layerwise_reps.append(img_rep[layer].flatten())
        layerwise_reps = np.array(layerwise_reps)
        normalized_layerwise_reps = normalize(layerwise_reps)
        
        van_indices = np.where(np.sum(normalized_layerwise_reps, axis=0) > 0)[0]
        normalized_layerwise_reps = normalized_layerwise_reps[:, van_indices]
        layer_re = []
        layer_si = []

        tetrad_count = 0
        for set_i in range(n_object_sets): #0-5
            set_start = set_i * n_stim_per_set
            for triplet_i in range(n_triplets): #0-3
                
                triplet_inds = (triplet_i * 3) + np.arange(1,4)
                img_inds = set_start + triplet_inds
        
                
                img1_resp = normalized_layerwise_reps[set_start]
                img2_resp = normalized_layerwise_reps[img_inds[0]]
                img3_resp = normalized_layerwise_reps[img_inds[1]]
                img4_resp = normalized_layerwise_reps[img_inds[2]]
      
            
                # identify active and inactive units for this tetrad
                temp_sum = np.array([img1_resp, img2_resp, img3_resp, img4_resp])
                temp_sum = np.sum(temp_sum, axis=0)

                active_tetrad_units = np.where(temp_sum > 0)[0]
                inactive_tetrad_units = np.where(temp_sum <= 0)[0]
                n_tetrads[layer_i, tetrad_count] = len(active_tetrad_units)


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
                layer_si.append(rel_size_index)

                tetrad_count += 1
            
        # select top 9%  tetrad units with highest residual error
        layer_re = np.array(layer_re).T #n_neuronsx24
        layer_si = np.array(layer_si).T #n_neuronsx24

        layer_re = layer_re.flatten()
        layer_si = layer_si.flatten()

        layer_re[np.isnan(layer_re)] = -9999999
        # sorted vals and inds 
        
        sorted_layer_inds = np.argsort(layer_re)[::-1]

    

        total_active_tetrads = np.sum(n_tetrads[layer_i])

        num_top_tetrads = int(np.floor(0.09*total_active_tetrads))


        selected_layer_si = layer_si[sorted_layer_inds[0:num_top_tetrads]]
        
        layer_si_score = np.nanmean(selected_layer_si)
        surface_invariance_index.append(layer_si_score)
        
        
        
    with open(f'./results/{model}_si.pkl', 'wb') as f:
        pickle.dump(surface_invariance_index, f)
    print("Saved surface_invariance index results for model: " + model)


for model in ['vgg16', 'vit_base']:
    get_surface_invariance_index(model)
    