import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from adjustText import adjust_text
import plotly.graph_objects as go

from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA
from sklearn.manifold import MDS

import os
import ast
import sys


sys.path.insert(1, 'D:/Niranjan_Work/dnns_qualitative/dnn_perception_expts')
from _utils.network import get_model_feature_layer

object_models = ["resnet50", "densenet161", "vit_base"]
face_models = ["facenet_casia", "facenet_vggface2", "face_vit"]
scene_models = [ "resnet50_places365", "densenet161_places365", "vit_base_places365"]

plot_models = {'Object': object_models, 'Face': face_models, 'Scene': scene_models}
total_expts = 17

benchmark_results_path = '../benchmark_results'

# 1 expt, all models
def plot_model_layerwise_scores(expt):
    plot_save_path = '../plots/datasets/layerwise' 
    if not os.path.exists(plot_save_path):
        os.makedirs(plot_save_path)
    # get 'brain score"
    brain_df = pd.read_csv(f'{benchmark_results_path}/brain/brain_proc_scores.csv', index_col=0)
    brain_score = dict(brain_df.iloc[expt-1])
    

   
    plt.figure(figsize=(10, 6))
    plt.axhspan(0, plt.gca().get_ylim()[1], color='gray', alpha=0.3) # Human perception zone

    colors =  ['#f8522e','#ba3e23', '#5d1f11', '#f8dd2e', '#f8c22e', '#f8a72e', '#2ed6f8', '#23a1ba', '#11505d']
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
        
        


    return brain_score

def load_model_scores(score_sel_method='features'):
    model_scores = {k: [0 for i in range(total_expts)] for k in object_models+face_models+scene_models} # 17d vector for each model (17 exps)
    expt_names = []
    expt_y_labels = []

    # TODO add more score_sel_methods (average, closest_to_brain, etc)
    for model_i, model in enumerate(object_models+face_models+scene_models):
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
                        if feat_layer_index == 1:
                            expt_model_score = 0
                
                # print(expt_model_score)
                model_scores[model][i] = float(expt_model_score)
            else:
                model_scores[model][i] = np.nan
           
            if model_i == 0:
                expt_name = model_scores_df.iloc[i]['expt_name']
                expt_y_label = model_scores_df.iloc[i]['expt_y_label']
                
                expt_names.append(expt_name)
                expt_y_labels.append(expt_y_label)
        
        
        

        

    return model_scores, (expt_names, expt_y_labels)


def expt_bar_plot(brain_scores, model_scores, expt_names):
    expt_titles, expt_y_labels = expt_names

    for expt_i in range(total_expts):
        if expt_i != 2:
            brain_score = brain_scores[expt_i]

            object_scores = {}
            face_scores = {}
            scene_scores = {}
            for model in object_models:
                object_scores[model] = model_scores[model][expt_i]
            for model in face_models:
                face_scores[model] = model_scores[model][expt_i]
            for model in scene_models:
                scene_scores[model] = model_scores[model][expt_i]

            all_scores = np.array(list(object_scores.values()) + list(face_scores.values()) + list(scene_scores.values()))
        
            # barplot of all scores and x axis labels of model names
            plt.figure(figsize=(12, 10))

            model_names = list(object_scores.keys()) + list(face_scores.keys()) + list(scene_scores.keys()) 
            colours =  ['#f8522e','#f8522e', '#f8522e', '#f8dd2e', '#f8dd2e', '#f8dd2e', '#2ed6f8', '#2ed6f8', '#2ed6f8']
            labels = ['Objects', 'Faces', 'Scenes']
            handles = [plt.Rectangle((0,0),1,1, color=colours[i]) for i in range(0,len(colours),3)]
            sns.barplot(x=model_names, y=all_scores, palette=colours)
            plt.legend(handles, labels, title='Training Data')
            # x ticks 45 deg
            plt.xticks(rotation=45)
            plt.title(expt_titles[expt_i])
            plt.axhline(y=brain_score, color='green', linestyle='--')
            plt.ylim(-1,1)
            if expt_i ==2:
                plt.ylabel('Score')
            else:
                plt.ylabel(expt_y_labels[expt_i])


            plot_path = '../plots/datasets/scores'
            expt_code = expt_titles[expt_i].split(':')[0]
            plt.savefig(f'{plot_path}/{expt_code}.png')
 
def brain_distance_plot(brain_scores, model_scores, dim_red='mds'):
        
        object_scores = {}
        face_scores = {}
        scene_scores = {}
        for model in object_models:
            object_scores[model] = model_scores[model]
        for model in face_models:
            face_scores[model] = model_scores[model]
        for model in scene_models:
            scene_scores[model] = model_scores[model]


        all_scores = np.array(list(object_scores.values()) + list(face_scores.values()) + list(scene_scores.values()) + [brain_scores])
       


        # expts_to_remove = [9,10]
        expts_to_remove = [3]
       

        for expt_i in expts_to_remove:
            new_scores  = [[] for m in all_scores] 
            for model_i in range(len(all_scores)):
                new_scores[model_i] = np.delete(all_scores[model_i], expt_i-1, axis=0)
            brain_scores[expt_i] = np.nan
            brain_scores = [ bs for bs in brain_scores if str(bs) != "nan" ]         # delete nans from brain_scores
            all_scores = new_scores
        
        if dim_red == 'mds':
            mds = MDS(n_components=2, dissimilarity='euclidean', random_state=0)
            coords = mds.fit_transform(all_scores)
        elif dim_red == 'pca':
            pca = PCA(n_components=2)
            coords = pca.fit_transform(all_scores)
        


        labels = list(model_scores.keys()) + ['Brain Score']
        colors =  ['#f8522e','#f8522e', '#f8522e', '#f8dd2e', '#f8dd2e', '#f8dd2e', '#2ed6f8', '#2ed6f8', '#2ed6f8', 'green']
        legend_labels = ['Objects', 'Faces', 'Scenes']
       
        handles = [plt.Rectangle((0,0),1,1, color=colors[i]) for i in range(0,len(colors),3)]

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
        plt.legend(handles, legend_labels, title='Training Regime')
        plot_path = '../plots/datasets/distances'
        plt.savefig(f'{plot_path}/all_expts_{dim_red}.png')
        # plt.show()

    
        
        
def compute_distance_from_brain(model_scores, brain_scores, metric='euclidean'):
    object_scores = {}
    face_scores = {}
    scene_scores = {}
    for model in object_models:
        object_scores[model] = model_scores[model]
    for model in face_models:
        face_scores[model] = model_scores[model]
    for model in scene_models:
        scene_scores[model] = model_scores[model]

    all_scores = np.array(list(object_scores.values()) + list(face_scores.values()) + list(scene_scores.values()))
    all_labels = list(object_scores.keys()) + list(face_scores.keys()) + list(scene_scores.keys())

    # expts_to_remove = [9,10]
    expts_to_remove = [3]


   
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
    colours =  ['#f8522e','#f8522e', '#f8522e', '#f8dd2e', '#f8dd2e', '#f8dd2e', '#2ed6f8', '#2ed6f8', '#2ed6f8']
    labels = ['Objects', 'Faces', 'Scenes']
    handles = [plt.Rectangle((0,0),1,1, color=colours[i]) for i in range(0,len(colours),3)]
    sns.barplot(x=all_labels, y=similarity, palette=colours)
    plt.legend(handles, labels, title='Training Regime')
    plt.title(f'Closeness to Brain Score across all experiments')
    plt.xlabel('Model')
    plt.ylabel('Brain similarity')
    plt.xticks(rotation=45)
    
    
    plot_path = '../plots/datasets/distances'
    plt.savefig(f'{plot_path}/all_brain_distances_{metric}.png')
    return
    
def model_plot_radar(brain_scores, model_scores, expt_names):
    object_scores = {}
    face_scores = {}
    scene_scores = {}
    for model in object_models:
        object_scores[model] = model_scores[model]
    for model in face_models:
        face_scores[model] = model_scores[model]
    for model in scene_models:
        scene_scores[model] = model_scores[model]

    

    all_scores = np.array(list(object_scores.values()) + list(face_scores.values()) + list(scene_scores.values()))
    all_models = list(object_scores.keys()) + list(face_scores.keys()) + list(scene_scores.keys())

  
    colors = [
        "#FF0000",
        "#FF0000",
        "#FF0000",
        "#FFD700",
        "#FFD700",
        "#FFD700",
        "#0055FF",
        "#0055FF",
        "#0055FF"
    ]
    dashed = ["solid", "solid", "dash", "solid", "solid", "dash", "solid", "dash", "solid"]

    expts_to_remove = [3]

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
            line={'color': colors[model_i], 'dash': dashed[model_i]},
            
        ))
        

    fig.add_trace(go.Scatterpolar(
        r=brain_scores,
        theta=expt_names,
        name='brain',
        line={'width': 5, 'color': 'green' },
    
    ))
    
    fig.update_layout(
    title=dict(
        text="Experiment scores for Networks trained on Different Datasets",
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
    
    fig.write_image(f'{benchmark_results_path}/datasets_radar.png')
    fig.show()
        
def main():
    # plot_layerwise_scores_all_expts()
    model_scores, expt_names = load_model_scores()   
    brain_scores = load_brain_score()
    

    # expt_bar_plot(brain_scores, model_scores, expt_names)
    # brain_distance_plot(brain_scores, model_scores, dim_red='mds')
    # brain_distance_plot(brain_scores, model_scores, dim_red='pca')
    # compute_distance_from_brain(model_scores, brain_scores)
    model_plot_radar(brain_scores, model_scores, expt_names)

if __name__ == "__main__":
    main()