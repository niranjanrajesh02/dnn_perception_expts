import sys
sys.path.insert(1, 'D:/Niranjan_Work/dnns_qualitative/dnn_perception_expts')
import numpy as np
import pickle
from tqdm import tqdm
from _utils.data import load_stim_file, save_stims
from _utils.network import load_model, get_layerwise_activations

stim_array = load_stim_file('./data/obj_parts_a_stim.mat', stim_var='img') 
print(stim_array.shape)


def get_part_matching_index(model):
    layers = load_model(model)
    print("Extracting layerwise activations...")
    image_reps = []
    for stim_i, stim in enumerate(tqdm(stim_array)):
        img_rep = get_layerwise_activations(stim)
        image_reps.append(img_rep)

    layerwise_part_matching = []

    print("Computing layerwise part matching...")
    for layer_i, layer in enumerate(tqdm(layers)):
        layerwise_reps = []
        
        for img_rep in image_reps:
            layerwise_reps.append(img_rep[layer].flatten())
        
        layerwise_reps = np.array(layerwise_reps)

        # SET 1
        normal_img = layerwise_reps[1]
        natural_cut = layerwise_reps[0]
        unnatural_cut = layerwise_reps[2]

        dn = np.linalg.norm(normal_img - natural_cut)
        du = np.linalg.norm(normal_img - unnatural_cut)

        part_matching1 = (du - dn) / (du + dn)

        # SET 2
        normal_img = layerwise_reps[4]
        natural_cut = layerwise_reps[3]
        unnatural_cut = layerwise_reps[5]

        dn = np.linalg.norm(normal_img - natural_cut)
        du = np.linalg.norm(normal_img - unnatural_cut)

        part_matching2 = (du - dn) / (du + dn)

        part_matching_index = (part_matching1 + part_matching2) / 2
        layerwise_part_matching.append(part_matching_index)
    
    with open(f'./results/{model}_part_matching.pkl', 'wb') as f:
        pickle.dump(layerwise_part_matching, f)
    print("Saved layerwise part matching")

for model in ['vgg16', 'vit_base']:
    get_part_matching_index(model)