import numpy as np
import pandas as pd
import sys
sys.path.insert(1, 'D:/Niranjan_Work/dnns_qualitative/dnn_perception_expts')
from _utils.network import get_model_feature_layer
import ast

from sklearn.manifold import MDS, TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances

import matplotlib.pyplot as plt
import seaborn as sns
from adjustText import adjust_text



benchmark_results_path = '../benchmark_results'

total_expts = 17 # includes sub-expts 
cnn_list = ['vgg16', 'vgg19', 'resnet50', 'resnet101', 'inception_v3', 'inception_v4', 'convnext_base', 'convnext_large']
vit_list = ['vit_base', 'vit_large', 'swin_base', 'swin_large', 'deit_base', 'deit_large']

# plot_models = {'CNNs': cnn_list, 'ViTs': vit_list}



def load_brain_score():
    brain_score = [0 for _ in range(total_expts)] # 17d vector 
    brain_df = pd.read_csv(f'{benchmark_results_path}/brain/brain_proc_scores.csv', index_col=0)

    for i in range(total_expts):

        if i!=2:
            brain_expt_score = dict(brain_df.iloc[i])['scores']
            brain_score[i] = float(brain_expt_score)

        else: #scene incog expt
            expt_score_dict = ast.literal_eval(dict(brain_df.iloc[i])['scores'])[0]
            # taking the mean of diff(congruent_acc, incogruent_acc)  of two diff studies
            brain_expt_score = np.mean([expt_score_dict['davenport_c_acc_mean']-expt_score_dict['davenport_i_acc_mean'], expt_score_dict['munneke_c_acc_mean']-expt_score_dict['munneke_i_acc_mean']])
            brain_score[i] = brain_expt_score

    return brain_score

def load_model_scores(score_sel_method='features'):
    model_scores = {k: [0 for i in range(total_expts)] for k in cnn_list+vit_list} # 17d vector for each model (17 exps)
    expt_names = []
    # TODO add more score_sel_methods (average, closest_to_brain, etc)
    for model in cnn_list+vit_list:

        model_results_path = f'{benchmark_results_path}/{model}'
        model_scores_df = pd.read_csv(f'{model_results_path}/{model}_proc_scores.csv', index_col=0)  # each row is an experiment

        if score_sel_method == 'features':
            feat_layer = get_model_feature_layer(model)

        for i in range(total_expts):
            # i=3
            if i != 2: #exclude scene incog expt 
                expt_model_score = model_scores_df.iloc[i][feat_layer]
                # handle nans
                if np.isnan(expt_model_score):
                    
                    feat_layer_index = model_scores_df.columns.get_loc(feat_layer)
                    while np.isnan(expt_model_score):
                        # if nan, go to previous layer
                        feat_layer_index -= 1
                        new_layer = model_scores_df.columns[feat_layer_index]
                        
                        expt_model_score = model_scores_df.iloc[i][feat_layer_index]
                
                
                # print(expt_model_score)
                model_scores[model][i] = float(expt_model_score)

            else: #scene incog expt
                expt_score_dict = ast.literal_eval(model_scores_df.iloc[i][2])[0]
                # * score = c_top1_acc - i_top1_acc
                expt_model_score = expt_score_dict['c_top1_acc_mean'] - expt_score_dict['i_top1_acc_mean']
                model_scores[model][i] = float(expt_model_score)
        
            expt_name = model_scores_df.iloc[i]['expt_name']
            expt_names.append(expt_name)
        
    return model_scores, expt_names
    


def brain_distance_plot(expt_i, brain_scores, model_scores, expt_names, dim_red='mds'):
    if expt_i == 'all':

        cnns_scores = {}
        vits_scores = {}
        for model in cnn_list:
            cnns_scores[model] = model_scores[model]
        for model in vit_list:
            vits_scores[model] = model_scores[model]

        all_scores = np.array(list(cnns_scores.values()) + list(vits_scores.values()) + [brain_scores])
        # print(all_scores)
        # print(all_scores.shape)

        if dim_red == 'mds':
            mds = MDS(n_components=2, dissimilarity='euclidean', random_state=0)
            coords = mds.fit_transform(all_scores)
        elif dim_red == 'pca':
            pca = PCA(n_components=2)
            coords = pca.fit_transform(all_scores)


        labels = list(model_scores.keys()) + ['Brain Score']
        colors = ['blue'] * len(cnns_scores) + ['orange'] * len(vits_scores) + ['green']

        plt.figure(figsize=(10, 8))

        scatter = plt.scatter(coords[:, 0], coords[:, 1], c=colors)
     
        texts = []
        for i, label in enumerate(labels):
            txt = plt.annotate(label, (coords[i, 0], coords[i, 1]), fontsize=10)
            texts.append(txt)

        adjust_text(texts, force_text=0.5, arrowprops=dict(arrowstyle='->', color='gray'))

        plt.title(f'Model Distance Plot for all experiments')
        plt.xlabel('MDS Dimension 1')
        plt.ylabel('MDS Dimension 2')
        plt.grid()
        plt.legend(handles=scatter.legend_elements()[0], labels=['CNNs', 'ViTs', 'Brain Score'], loc='upper right')
        plot_path = '../plots/distances'
        plt.savefig(f'{plot_path}/all_expts.png')
        # plt.show()

    else:
        expt_i = int(expt_i)
        brain_score = brain_scores[expt_i]

        cnns_scores = {}
        vits_scores = {}
        for model in cnn_list:
            cnns_scores[model] = model_scores[model][expt_i]
        for model in vit_list:
            vits_scores[model] = model_scores[model][expt_i]
        
        all_scores = np.array(list(cnns_scores.values()) + list(vits_scores.values()) + [brain_score])
        all_scores = all_scores.reshape(-1, 1) #2d for the paiwise_distances func
        # print(all_scores)

        mds = MDS(n_components=2, dissimilarity='euclidean', random_state=0)
        coords = mds.fit_transform(all_scores)

    
        labels = list(cnns_scores.keys()) + list(vits_scores.keys()) + ['Brain Score']
        colors = ['blue'] * len(cnns_scores) + ['orange'] * len(vits_scores) + ['green']

        
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(coords[:, 0], coords[:, 1], c=colors, alpha=0.7)
        x_range = np.max(coords[:, 0]) - np.min(coords[:, 0])
        y_range = np.max(coords[:, 1]) - np.min(coords[:, 1])

        texts = []
        for i, label in enumerate(labels):
            txt = plt.annotate(label, (coords[i, 0], coords[i, 1]), fontsize=10)
            texts.append(txt)

        adjust_text(texts, force_text=0.5, arrowprops=dict(arrowstyle='->', color='gray'))

        plt.title(f'Model Distance Plot for {expt_names[expt_i]}')
        plt.xlabel('MDS Dimension 1')
        plt.ylabel('MDS Dimension 2')
        plt.grid()
        plt.legend(handles=scatter.legend_elements()[0], labels=['CNNs', 'ViTs', 'Brain Score'], loc='upper right')
        plot_path = '../plots/distances'
        expt_code = expt_names[expt_i].split(':')[0]
        plt.savefig(f'{plot_path}/{expt_code}.png')
        
        
def compute_distance_from_brain(model_scores, brain_scores, metric='euclidean'):
    cnns_scores = {}
    vits_scores = {}
    for model in cnn_list:
        cnns_scores[model] = model_scores[model]
    for model in vit_list:
        vits_scores[model] = model_scores[model]

    all_scores = np.array(list(cnns_scores.values()) + list(vits_scores.values()))
    all_labels = list(cnns_scores.keys()) + list(vits_scores.keys())
    colours = ['blue'] * len(cnns_scores) + ['orange'] * len(vits_scores)
    
    if metric == 'euclidean':
        distances_from_brain = pairwise_distances(all_scores, [brain_scores]).flatten()
    elif metric == 'cosine':
        distances_from_brain = pairwise_distances(all_scores, [brain_scores], metric='cosine').flatten()
    elif metric == 'manhattan':
        distances_from_brain = pairwise_distances(all_scores, [brain_scores], metric='manhattan').flatten()

    plt.figure(figsize=(10, 8))

    sns.barplot(x=all_labels, y=distances_from_brain, palette=colours)

    plt.title(f'Distance from Brain')
    plt.xlabel('Model')
    plt.ylabel('Distance')
    plt.xticks(rotation=45)
    
    plot_path = '../plots/distances'
    plt.savefig(f'{plot_path}/all_brain_distances_{metric}.png')
    pass


def main():
    model_scores, expt_names = load_model_scores()   
    brain_scores = load_brain_score()
    # for i in tqdm(range(total_expts)):
    #     brain_distance_plot(i, brain_scores, model_scores, expt_names)

    # brain_distance_plot('all', brain_scores, model_scores, expt_names)
    compute_distance_from_brain(model_scores, brain_scores, metric='euclidean')
    compute_distance_from_brain(model_scores, brain_scores, metric='cosine')
    compute_distance_from_brain(model_scores, brain_scores, metric='manhattan')

    
   
    

if __name__ == "__main__":

    main()
    # load_brain_score()
