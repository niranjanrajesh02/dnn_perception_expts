import sys
sys.path.insert(1, 'D:/Niranjan_Work/dnns_qualitative/dnn_perception_expts')
import numpy as np
import pickle
from tqdm import tqdm
from _utils.data import load_stim_file, save_stims
from _utils.network import load_model, get_layerwise_activations
from scipy.io import loadmat
from sklearn.metrics import pairwise_distances

stim_array = load_stim_file('./data/GL.mat')
# print(stim_array.shape)# 49x3x224x224

num_shapes = 7 # ( 7 shapes with same local shape but varying global shape ) x 7 times for a local shape
# global groups: 0 - diam, 1 - square, 2 - A, 3 - Circle, 4 - X, 5 - N, 6 - Z

img_pair_inds = loadmat('./data/img_pair_inds.mat')['imagepairDetails']
g1, l1 = img_pair_inds[:,0], img_pair_inds[:,1]
g2, l2 = img_pair_inds[:,2], img_pair_inds[:,3]

indexG = np.where(l1==l2)[0]    
indexL = np.where(g1==g2)[0]


global_diff_local_same_inds = [] #global_inds

for g in range(7):
    for i in range(7):
        for j in range(i+1,7):
            global_diff_local_same_inds.append([g*7+i,g*7+j])





# # extract stim features
def get_global_advantage(model):
    layers = load_model(model)
    print("Extracting layerwise activations...")
    image_reps = []
    for stim_i, stim in enumerate(tqdm(stim_array)):
        img_rep = get_layerwise_activations(stim)
        image_reps.append(img_rep)

    layerwise_global_advantage = []

    print("Computing layerwise global advantage...")
    for layer_i, layer in enumerate(tqdm(layers)):
        layerwise_reps = []
        
        for img_rep in image_reps:
            layerwise_reps.append(img_rep[layer].flatten())

        layerwise_reps = np.array(layerwise_reps)
        
        layerwise_dist = pairwise_distances(layerwise_reps, metric='euclidean') # matrix of pairwise distances between all stimuli
        layerwise_dist = layerwise_dist[np.triu_indices(layerwise_dist.shape[0], k=1)]
                    
        
        
        mean_global_dist = np.nanmean(layerwise_dist[indexG])
        mean_local_dist = np.nanmean(layerwise_dist[indexL])

        global_advantage = (mean_global_dist - mean_local_dist) / (mean_global_dist + mean_local_dist)
        layerwise_global_advantage.append(global_advantage)

    with open(f'./results/{model}_global_adv.pkl', 'wb') as f:
        pickle.dump(layerwise_global_advantage, f)

for model in ['vgg16', 'vit_base']:
    get_global_advantage(model)