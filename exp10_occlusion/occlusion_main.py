import sys
sys.path.insert(1, 'D:/Niranjan_Work/dnns_qualitative/dnn_perception_expts')
import numpy as np
import pickle
from tqdm import tqdm
from _utils.data import load_stim_file, save_stims
from _utils.network import load_model, get_layerwise_activations

stim_array = load_stim_file('./data/occlusion_set.mat') #14,4,224,224
# ordering is a little arbitrary, save these images using _utils.data.save_stims to visualise the indices of the images


def get_occlusion_scores(model):

    layers = load_model(model)
    img_reps = []
    print("Extracting layerwise activations...")
    for stim_i, stim in enumerate(tqdm(stim_array)):
        img_rep = get_layerwise_activations(stim)
        img_reps.append(img_rep)
    img_reps = np.array(img_reps)

    basic_occlusion_scores = []
    depth_ordering_occlusion_scores = []

    print("Computing layerwise occlusion scores...")
    for layer_i, layer in enumerate(tqdm(layers)):
        layerwise_reps = []

        for img_rep in img_reps:
            layerwise_reps.append(img_rep[layer].flatten())

        layer_basic_occlusion = []
        # basic occlusion 
        for set_i in range(2): # there are two sets (flipped versions)
            img1_reps = layerwise_reps[set_i*3 + 0] #unoccluded
            img2_reps = layerwise_reps[set_i*3 + 1] #occluded
            img3_reps = layerwise_reps[set_i*3 + 2] #equivalent feature distance

            dist_12 = np.linalg.norm(img1_reps - img2_reps) #distance between unoccluded and occluded (low in humans)
            dist_13 = np.linalg.norm(img1_reps - img3_reps) #distance between unoccluded and equivalent (high in humans)
        
            basic_occlusion_index = (dist_13 - dist_12) / (dist_13 + dist_12)
            layer_basic_occlusion.append(basic_occlusion_index)
        
        layer_basic_occlusion_score = np.mean(layer_basic_occlusion)
        basic_occlusion_scores.append(layer_basic_occlusion_score)
        
        # depth occlusion
        layer_depth_occlusion = []
        # set 1
        occluded_reps = layerwise_reps[6]
        flipped_occluded_reps = layerwise_reps[1]
        equivalent_reps = layerwise_reps[7]

        dist2 = np.linalg.norm(occluded_reps - equivalent_reps) #should be higher
        dist1 = np.linalg.norm(occluded_reps - flipped_occluded_reps)

        depth_occlusion_index = (dist2 - dist1) / (dist2 + dist1)
        layer_depth_occlusion.append(depth_occlusion_index)

        # set 2 (flipped 1)
        occluded_reps = layerwise_reps[8]
        flipped_occluded_reps = layerwise_reps[4]
        equivalent_reps = layerwise_reps[9]
        dist2 = np.linalg.norm(occluded_reps - equivalent_reps)
        dist1 = np.linalg.norm(occluded_reps - flipped_occluded_reps)

        depth_occlusion_index = (dist2 - dist1) / (dist2 + dist1)
        layer_depth_occlusion.append(depth_occlusion_index)

        # set 3 (inverse of 1)
        occluded_reps = layerwise_reps[1]
        flipped_occluded_reps = layerwise_reps[6]
        equivalent_reps = layerwise_reps[10]

        dist2 = np.linalg.norm(occluded_reps - equivalent_reps)
        dist1 = np.linalg.norm(occluded_reps - flipped_occluded_reps)

        depth_occlusion_index = (dist2 - dist1) / (dist2 + dist1)
        layer_depth_occlusion.append(depth_occlusion_index)

        # set 4 (flipped 3)
        occluded_reps = layerwise_reps[4]
        flipped_occluded_reps = layerwise_reps[8]
        equivalent_reps = layerwise_reps[11]

        dist2 = np.linalg.norm(occluded_reps - equivalent_reps)
        dist1 = np.linalg.norm(occluded_reps - flipped_occluded_reps)

        depth_occlusion_index = (dist2 - dist1) / (dist2 + dist1)
        layer_depth_occlusion.append(depth_occlusion_index)

        
        layer_depth_occlusion_score = np.mean(layer_depth_occlusion)
        depth_ordering_occlusion_scores.append(layer_depth_occlusion_score)

    occlusion_scores = {'basic': basic_occlusion_scores, 'depth': depth_ordering_occlusion_scores}

    with open(f'./results/{model}_occlusion_scores.pkl', 'wb') as f:
        pickle.dump(occlusion_scores, f)
    print("Saved Occlusion Scores!")

for model in ['vgg16', 'vit_base']:
    get_occlusion_scores(model)
   

