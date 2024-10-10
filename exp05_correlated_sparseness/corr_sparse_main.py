import sys
sys.path.insert(1, 'D:/Niranjan_Work/dnns_qualitative/dnn_perception_expts')
import numpy as np
import pickle
from tqdm import tqdm
from _utils.data import load_stim_file, save_stims
from _utils.network import load_model, get_layerwise_activations
import matplotlib.pyplot as plt
from _utils.stats import nan_corrcoeff, normalize

def sparseness(r, normalize=True):
    # input: r => n_units x n_stimuli
    # output: s => n_units x 1
    
    
    
    n = r.shape[1]  # total number of stimuli (number of columns)

    # Calculate sparseness
    s = (1 - (1 / n) * (np.sum(r, axis=1) ** 2) / np.sum(r ** 2, axis=1)) / (1 - 1 / n)
    
    # Calculate effective number of stimuli
    k = n - (n - 1) * s

    return s, k






def morph_ref_sparseness_correlation(model):
    morph_stim_data = load_stim_file('./data/seltol.mat') #116,3,224,224
    morph_stim_data = morph_stim_data[:44] # 44,3,224,224 (only first 44 are relevant)
    n_stim = 44
    n_morphs = 4
    ref_indices = [0,11,22,33,10,21,32,43] #reference images
    morph_indices = range(n_stim)

    image_reps = []
    layers = load_model(model)

    # extract activations
    print("Extracting layerwise activations...")
    for stim_i in tqdm(range(n_stim)):
        img_rep = get_layerwise_activations(morph_stim_data[stim_i])
        image_reps.append(img_rep)

    print("Computing layerwise sparseness correlation...")
    layerwise_sparseness_correlation = []
 
    for layer_i, layer in enumerate(tqdm(layers)):
        layerwise_reps = []
        
        for img_rep in image_reps:
            layerwise_reps.append(img_rep[layer].flatten())
        layerwise_reps = np.array(layerwise_reps) # n_stim x n_units
        normalized_layerwise_reps = normalize(layerwise_reps)

        van_units = np.where(np.sum(normalized_layerwise_reps, axis=0)>0)[0] # if one unit responds the same way to all stimuli, its' normalized value will be 0, so sum will also remain 0
        layerwise_van_reps = normalized_layerwise_reps[:, van_units]


        # # find visually active units
        # var_reps = np.var(layerwise_reps, axis=0)
        # van_units = np.where(var_reps > 0.01)[0] # visually active neurons => their activations vary across stim
        # layerwise_van_reps = layerwise_reps[:, van_units]
        
        # finding sparseness on ref set
        ref_sparseness = sparseness(layerwise_van_reps[ref_indices, :].T)[0] # find sparseness of all units across ref
        
        morph_sparseness = []
        for i in range(n_morphs):
            morph_sparseness.append(sparseness(layerwise_van_reps[morph_indices[11*(i):11*(i)+11], :].T)[0])

        # find max morph sparseness
        morph_sparseness = np.array(morph_sparseness)
        max_morph_sparseness = morph_sparseness.max(axis=0)
        ref_sparseness = np.array(ref_sparseness)

        corr = nan_corrcoeff(ref_sparseness, max_morph_sparseness)
        layerwise_sparseness_correlation.append(corr[0])
        
        

    with open(f'./results/morph_ref_{model}.pkl', 'wb') as f:
        pickle.dump(layerwise_sparseness_correlation, f)
    print("Saved Layerwise Sparseness Correlation Scores!")


def shape_texture_sparseness_correlation(model):
    shapes_stim_data = load_stim_file('./data/shapes.mat') 
    textures_stim_data = load_stim_file('./data/textures.mat')
    n_stim = 50
    shapes_stim_data = shapes_stim_data[:n_stim]
    textures_stim_data = textures_stim_data[:n_stim]
    stim = np.concatenate((shapes_stim_data, textures_stim_data), axis=0)
    
    image_reps = []
    layers = load_model(model)

    # extract activations
    print("Extracting layerwise activations...")
    for stim_i in tqdm(range(n_stim*2)):
        img_rep = get_layerwise_activations(stim[stim_i])
        image_reps.append(img_rep)
    
    print("Computing layerwise sparseness correlation...")
    layerwise_sparseness_correlation = []
    for layer_i, layer in enumerate(tqdm(layers)):
        layerwise_reps = []
        
        for img_rep in image_reps:
            layerwise_reps.append(img_rep[layer].flatten())

        layerwise_reps = np.array(layerwise_reps)
        normalized_layerwise_reps = normalize(layerwise_reps)
        normalized_layerwise_shape_reps = normalized_layerwise_reps[:n_stim]
        normalized_layerwise_texture_reps = normalized_layerwise_reps[n_stim:]

    

        van_units_shape = np.where(np.sum(normalized_layerwise_shape_reps, axis=0)>0)[0]
        van_units_texture = np.where(np.sum(normalized_layerwise_texture_reps, axis=0)>0)[0]

        van_units_both = np.intersect1d(van_units_shape, van_units_texture)
        

        # find visually active units for both sets
        # var_reps_shape = np.var(layerwise_shape_reps, axis=0)
        # var_reps_texture = np.var(layerwise_texture_reps, axis=0)
        # van_units_shape = np.where(var_reps_shape > 0.01)[0] # visually active neurons => their activations vary across stim
        # van_units_texture = np.where(var_reps_texture > 0.01)[0] # visually active neurons => their activations vary across stim
        # # find units selective for both shape and texture
        # van_units_both = np.intersect1d(van_units_shape, van_units_texture)
        
        layerwise_van_reps_shape = normalized_layerwise_shape_reps[:, van_units_both]
        layerwise_van_reps_texture = normalized_layerwise_texture_reps[:, van_units_both]
        
        # finding sparseness 
        shape_sparseness = sparseness(layerwise_van_reps_shape.T)[0] 
        texture_sparseness = sparseness(layerwise_van_reps_texture.T)[0]

        corr = nan_corrcoeff(shape_sparseness, texture_sparseness)
        layerwise_sparseness_correlation.append(corr[0])
        
    
    with open(f'./results/shape_texture_{model}.pkl', 'wb') as f:
        pickle.dump(layerwise_sparseness_correlation, f)
    print("Saved Layerwise Sparseness Correlation Scores!")

for model in ['vgg16', 'vit_base']:
    morph_ref_sparseness_correlation(model)
    shape_texture_sparseness_correlation(model)
# might need to run multiple times to get stable results

# morph_ref_sparseness_correlation('vgg16')
# shape_texture_sparseness_correlation('vgg16')