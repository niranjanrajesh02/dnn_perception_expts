import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from adjustText import adjust_text
import plotly.graph_objects as go

from sklearn.metrics import pairwise_distances
from sklearn.manifold import MDS
from sklearn.decomposition import PCA

import os
import ast
import sys


sys.path.insert(1, 'D:/Niranjan_Work/dnns_qualitative/dnn_perception_expts')
from _utils.network import get_model_feature_layer

supervised_list = ['resnet50', 'vit_base', 'resnet50_at', 'vit_base_at']
self_supervised_list = ['resnet50_moco', 'vit_base_moco', 'resnet50_dino', 'vit_base_dino']

plot_models = {'Supervised': supervised_list, 'Self-Supervised': self_supervised_list}

total_expts = 17

benchmark_results_path = '../benchmark_results'

# 1 expt, all models
def plot_model_layerwise_scores(expt):
    plot_save_path = '../plots/training_reg/layerwise' 
    if not os.path.exists(plot_save_path):
        os.makedirs(plot_save_path)
    # get 'brain score"
    brain_df = pd.read_csv(f'{benchmark_results_path}/brain/brain_proc_scores.csv', index_col=0)
    brain_score = dict(brain_df.iloc[expt-1])
    

   
    plt.figure(figsize=(10, 6))
    plt.axhspan(0, plt.gca().get_ylim()[1], color='gray', alpha=0.3) # Human perception zone

    colors =  ['#ff5733','#ffbd33', '#bf3213', '#bf8813', '#3375ff', '#33dbff', '#0d3da0', '#0d86a0']
    line_index = 0
    for model_family, model_list in plot_models.items():
        for model in model_list:
            model_results_path = f'{benchmark_results_path}/{model}'
            model_scores_df = pd.read_csv(f'{model_results_path}/{model}_proc_scores.csv', index_col=0)  # each row is an experiment
            # model_sem_df = pd.read_csv(f'{model_results_path}/{model}_sem.csv', index_col=0)  # each row is an experiment

            model_layerwise_score = dict(model_scores_df.iloc[expt-1])
            # remove the first two columns after reading
            model_layerwise_score = {key: value for key, value in model_layerwise_score.items() if key != 'expt_name' and key != 'expt_y_label'} #exclude first two columns

            # string vals to float unless they are nans
            model_layerwise_score = {k: float(v) for k, v in model_layerwise_score.items()}
            
            # normalise layers and scores
            num_layers = len(model_layerwise_score)
            
            normalised_layer_scores = []
            # normalise scores to new range
            
           
            layer_indices = np.linspace(0, 100, num_layers) # x values: 0-100, num_layers points
            normalised_layer_scores = list(model_layerwise_score.values())
  
            sns.lineplot(x=layer_indices, y=normalised_layer_scores, label=model, color=colors[line_index])
            line_index +=1
    
        expt_title = dict(model_scores_df.iloc[expt-1])['expt_name']
        expt_y_label = dict(model_scores_df.iloc[expt-1])['expt_y_label']    

        
        ref_label = brain_score['reference']
        ref_val = float(brain_score['scores'])
        label_pos = 0.01

    if expt == 4 or expt ==5:
        label_pos = -0.1
    plt.axhline(y=ref_val, xmin=0, xmax=100, color='black', linestyle='--')
    plt.text(5, ref_val+label_pos, ref_label, fontsize=12, color='black', ha='left', va='bottom')
    plt.title(f'{expt_title}')
    plt.xlabel('Model Depth (%)')
    plt.ylabel(expt_y_label)
    plt.ylim(-1, 1)
    plt.xlim(0,100)
    plt.legend(title="Network")
    expt_code = expt_title.split(':')[0]
    plt.savefig(f'{plot_save_path}/{expt_code}.png')
    plt.close()

def plot_layerwise_scores_all_expts():
    for expt in range(1,18):
        if expt != 3:
            plot_model_layerwise_scores(expt)


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
            davenport_score = (expt_score_dict['davenport_c_acc_mean']-expt_score_dict['davenport_i_acc_mean']) / (expt_score_dict['davenport_c_acc_mean']+expt_score_dict['davenport_i_acc_mean'])
            munneke_score = (expt_score_dict['munneke_c_acc_mean']-expt_score_dict['munneke_i_acc_mean']) / (expt_score_dict['munneke_c_acc_mean']+expt_score_dict['munneke_i_acc_mean'])
            brain_expt_score = np.mean([davenport_score, munneke_score])
            brain_score[i] = brain_expt_score

    return brain_score

def load_model_scores(score_sel_method='features'):
    model_scores = {k: [0 for i in range(total_expts)] for k in supervised_list+self_supervised_list} # 17d vector for each model (17 exps)
    expt_names = []
    expt_y_labels = []

    # TODO add more score_sel_methods (average, closest_to_brain, etc)
    for model_i, model in enumerate(supervised_list+self_supervised_list):
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
                        
                        expt_model_score = model_scores_df.iloc[i][feat_layer_index]
                
                # print(expt_model_score)
                model_scores[model][i] = float(expt_model_score)

            else: #scene incog expt
                expt_score_dict = ast.literal_eval(model_scores_df.iloc[i][2])[0]
                # * score = c_top1_acc - i_top1_acc
                expt_model_score = (expt_score_dict['c_top1_acc_mean'] - expt_score_dict['i_top1_acc_mean']) / (expt_score_dict['c_top1_acc_mean'] + expt_score_dict['i_top1_acc_mean'])
                model_scores[model][i] = float(expt_model_score)

            if model_i == 0:
                expt_name = model_scores_df.iloc[i]['expt_name']
                expt_y_label = model_scores_df.iloc[i]['expt_y_label']
                
                expt_names.append(expt_name)
                expt_y_labels.append(expt_y_label)
    
    return model_scores, (expt_names, expt_y_labels)


def expt_bar_plot(brain_scores, model_scores, expt_names):
    expt_titles, expt_y_labels = expt_names

    for expt_i in range(total_expts):
        brain_score = brain_scores[expt_i]

        supervised_scores = {}
        self_supervised_scores = {}
        for model in supervised_list:
            supervised_scores[model] = model_scores[model][expt_i]
        for model in self_supervised_list:
            self_supervised_scores[model] = model_scores[model][expt_i]

        all_scores = np.array(list(supervised_scores.values()) + list(self_supervised_scores.values()))
     
        # barplot of all scores and x axis labels of model names
        plt.figure(figsize=(12, 10))

        model_names = list(supervised_scores.keys()) + list(self_supervised_scores.keys()) 
        colours =  ['#ff5733','#ff5733', '#bf3213', '#bf3213', '#3375ff', '#3375ff', '#0d3da0', '#0d3da0']
        labels = ['Standard', 'Adversarial', 'MOCOv2', 'DINO']
        handles = [plt.Rectangle((0,0),1,1, color=colours[i]) for i in range(0,len(colours),2)]
        sns.barplot(x=model_names, y=all_scores, palette=colours)
        plt.legend(handles, labels, title='Training Regime')
        # x ticks 45 deg
        plt.xticks(rotation=45)
        plt.title(expt_titles[expt_i])
        plt.axhline(y=brain_score, color='green', linestyle='--')
        plt.ylim(-1,1)
        if expt_i ==2:
            plt.ylabel('Score')
        else:
            plt.ylabel(expt_y_labels[expt_i])


        plot_path = '../plots/training_reg/scores'
        expt_code = expt_titles[expt_i].split(':')[0]
        plt.savefig(f'{plot_path}/{expt_code}.png')
 
def brain_distance_plot(brain_scores, model_scores, dim_red='mds'):
        supervised_scores = {}
        self_supervised_scores = {}
        for model in supervised_list:
            supervised_scores[model] = model_scores[model]
        for model in self_supervised_list:
            self_supervised_scores[model] = model_scores[model]

        all_scores = np.array(list(supervised_scores.values()) + list(self_supervised_scores.values()) + [brain_scores])
        # print(all_scores)
        # print(all_scores.shape)

        # expts_to_remove = [9,10]
        expts_to_remove = []

        for expt_i in expts_to_remove:
            new_scores  = [[] for m in all_scores] 
            for model_i in range(len(all_scores)):
                new_scores[model_i] = np.delete(all_scores[model_i], expt_i-1, axis=0)
            brain_scores[expt_i] = np.nan
            all_scores = new_scores
        
        # delete nans from brain_scores
        brain_scores = [ bs for bs in brain_scores if str(bs) != "nan" ]

        if dim_red == 'mds':
            mds = MDS(n_components=2, dissimilarity='euclidean', random_state=0)
            coords = mds.fit_transform(all_scores)
        elif dim_red == 'pca':
            pca = PCA(n_components=2)
            coords = pca.fit_transform(all_scores)
       



        labels = list(model_scores.keys()) + ['Brain Score']
        colors =  ['#ff5733','#ff5733', '#bf3213', '#bf3213', '#3375ff', '#3375ff', '#0d3da0', '#0d3da0', 'green']
        legend_labels = ['Standard', 'Adversarial', 'MOCOv2', 'DINO', 'Brain Score']
        handles = [plt.Rectangle((0,0),1,1, color=colors[i]) for i in range(0,len(colors),2)]

        plt.figure(figsize=(10, 8))

        scatter = plt.scatter(coords[:, 0], coords[:, 1], c=colors)
     
        texts = []
        for i, label in enumerate(labels):
            txt = plt.annotate(label, (coords[i, 0], coords[i, 1]), fontsize=10)
            texts.append(txt)

        adjust_text(texts, force_text=0.5, arrowprops=dict(arrowstyle='->', color='gray'))

        plt.title(f'Model Distance Plot for all experiments')
        plt.xlabel(f'{dim_red.upper()} Dimension 1')
        plt.ylabel(f'{dim_red.upper()} Dimension 2')
        plt.grid()
        plt.legend(handles, legend_labels, title='Training Regime')
        plot_path = '../plots/training_reg/distances'
        plt.savefig(f'{plot_path}/all_expts_{dim_red}.png')
        # plt.show()
      
        
def compute_distance_from_brain(model_scores, brain_scores, metric='euclidean'):
    supervised_scores = {}
    self_supervised_scores = {}
    for model in supervised_list:
        supervised_scores[model] = model_scores[model]
    for model in self_supervised_list:
        self_supervised_scores[model] = model_scores[model]

    all_scores = np.array(list(supervised_scores.values()) + list(self_supervised_scores.values()))
    all_labels = list(supervised_scores.keys()) + list(self_supervised_scores.keys())
    

    # expts_to_remove = [9,10]
    expts_to_remove = []


   
    for expt_i in expts_to_remove:
        new_scores  = [[] for m in all_labels] 
        for model_i in range(len(all_scores)):
            new_scores[model_i] = np.delete(all_scores[model_i], expt_i-1, axis=0)
        brain_scores[expt_i] = np.nan
        all_scores = new_scores
    
    # delete nans from brain_scores
    brain_scores = [ bs for bs in brain_scores if str(bs) != "nan" ]
    
    
    if metric == 'euclidean':
        distances_from_brain = pairwise_distances(all_scores, [brain_scores]).flatten()
    elif metric == 'cosine':
        distances_from_brain = pairwise_distances(all_scores, [brain_scores], metric='cosine').flatten()
    elif metric == 'manhattan':
        distances_from_brain = pairwise_distances(all_scores, [brain_scores], metric='manhattan').flatten()

    # print(distances_from_brain)
    similarity = 1 / (1+distances_from_brain)
    plt.figure(figsize=(14, 12))
    colours =  ['#ff5733','#ff5733', '#bf3213', '#bf3213', '#3375ff', '#3375ff', '#0d3da0', '#0d3da0']
    labels = ['Standard', 'Adversarial', 'MOCOv2', 'DINO']
    handles = [plt.Rectangle((0,0),1,1, color=colours[i]) for i in range(0,len(colours),2)]
    sns.barplot(x=all_labels, y=similarity, palette=colours)
    plt.legend(handles, labels, title='Training Regime')
    plt.title(f'Closeness to Brain Score across all experiments')
    plt.xlabel('Model')
    plt.ylabel('Brain similarity')
    plt.xticks(rotation=45)
    
    
    plot_path = '../plots/training_reg/distances'
    plt.savefig(f'{plot_path}/all_brain_distances_{metric}.png')
    return


def model_plot_radar(brain_scores, model_scores, expt_names):
    supervised_scores = {}
    self_supervised_scores = {}
    for model in supervised_list:
        supervised_scores[model] = model_scores[model]
    for model in self_supervised_list:
        self_supervised_scores[model] = model_scores[model]

    all_scores = np.array(list(supervised_scores.values()) + list(self_supervised_scores.values()))
    all_models = list(supervised_scores.keys()) + list(self_supervised_scores.keys())

  
    colors = [
    "#FF0000",
    "#FF0000",
    "#FFA500",
    "#FFA500",
    "#0000FF",
    "#0000FF",
    "#00008B",
    "#00008B"
]
    dashed = ["solid", "dash", "solid", "dash", "solid", "dash", "solid", "dash"]

    expts_to_remove = []

    for expt_i in expts_to_remove:
        new_scores  = [[] for m in all_scores] 
        for model_i in range(len(all_scores)):
            new_scores[model_i] = np.delete(all_scores[model_i], expt_i-1, axis=0)
        brain_scores[expt_i] = np.nan
        brain_scores = [ bs for bs in brain_scores if str(bs) != "nan" ]         # delete nans from brain_scores
        expt_names = np.delete(expt_names, expt_i-1)
        all_scores = new_scores
    
    fig = go.Figure()
    for model_i, model in enumerate(all_models):
        scores = all_scores[model_i] 
        labels = expt_names 

        fig.add_trace(go.Scatterpolar(
            r=scores,
            theta=labels,
            name=f'{model}',
            line={'color': colors[model_i], 'dash':dashed[model_i] },
            
        ))
        

    fig.add_trace(go.Scatterpolar(
        r=brain_scores,
        theta=expt_names,
        name='brain',
        line={'width': 5, 'color': 'green' },
    
    ))
    
    fig.update_layout(
    title=dict(
        text="Experiment scores for Networks with different Training Regimes",
        x=0.5,
        xanchor="center",
        font=dict(size=18)
    ),
    polar=dict(
        angularaxis=dict(
            tickfont=dict(size=12),
        ),
        radialaxis=dict(
        visible=True,
        range=[-1, 1],
        tickvals=[-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1],  # Grid every 0.2
        ticktext=["-1", "", "", "", "", "0", "", "", "", "", "1"],  # Labels only for -1, 0, 1
        )),
    showlegend=True,
    width=1400,
    height=900
    )
    
    fig.write_image(f'{benchmark_results_path}/training_reg_radar.png')
    fig.show()

def main():
    # plot_layerwise_scores_all_expts()
    model_scores, expt_names = load_model_scores()   
    brain_scores = load_brain_score()
    # expt_bar_plot(brain_scores, model_scores, expt_names)
    # compute_distance_from_brain(model_scores, brain_scores)
    brain_distance_plot(brain_scores, model_scores, dim_red='mds')
    brain_distance_plot(brain_scores, model_scores, dim_red='pca')
    model_plot_radar(brain_scores, model_scores, expt_names[0])

if __name__ == "__main__":
    main()