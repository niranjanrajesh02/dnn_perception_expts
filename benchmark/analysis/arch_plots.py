import numpy as np
import pandas as pd
import sys
sys.path.insert(1, 'D:/Niranjan_Work/dnns_qualitative/dnn_perception_expts')
from _utils.network import get_model_feature_layer
import ast
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import pandas as pd

benchmark_results_path = '../benchmark_results'

total_expts = 17 # includes sub-expts 
cnn_list = ['vgg16', 'vgg19', 'resnet50', 'resnet101', 'inception_v3', 'inception_v4', 'convnext_base', 'convnext_large']
vit_list = ['vit_base', 'vit_large', 'swin_base', 'swin_large', 'deit_base', 'deit_large']


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
    model_scores = {k: [0 for i in range(total_expts)] for k in cnn_list+vit_list} # 17d vector for each model (17 exps)
    expt_names = []
    expt_y_labels = []
    # TODO add more score_sel_methods (average, closest_to_brain, etc)
    for model_i, model in enumerate(cnn_list+vit_list):

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
                expt_model_score = (expt_score_dict['c_top1_acc_mean'] - expt_score_dict['i_top1_acc_mean']) / (expt_score_dict['c_top1_acc_mean'] + expt_score_dict['i_top1_acc_mean'])
                model_scores[model][i] = float(expt_model_score)

            if model_i == 0:
                expt_name = model_scores_df.iloc[i]['expt_name']
                expt_y_label = model_scores_df.iloc[i]['expt_y_label']
                
                expt_names.append(expt_name)
                expt_y_labels.append(expt_y_label)
    
    return model_scores, (expt_names, expt_y_labels)


def model_plot_radar(brain_scores, model_scores, expt_names):
    cnns_scores = {}
    vits_scores = {}
    for model in cnn_list:
        cnns_scores[model] = model_scores[model]
    for model in vit_list:
        vits_scores[model] = model_scores[model]

    all_scores = np.array(list(cnns_scores.values()) + list(vits_scores.values()))
    all_models = list(cnns_scores.keys()) + list(vits_scores.keys())

  
    fig = go.Figure()
    colors = [
        "#FF6666", "#FF4D4D", "#FF3333", "#FF1A1A", "#FF0000", "#E60000", "#CC0000", "#B20000",
        "#6699FF", "#4D88FF", "#3377FF", "#1A66FF", "#0055FF", "#0044CC"
        ]
    
    for model_i, model in enumerate(all_models):
        scores = all_scores[model_i] 
        labels = expt_names 
     
        fig.add_trace(go.Scatterpolar(
            r=scores,
            theta=labels,
            name=f'{model}',
            line={'color': colors[model_i]},
            
        ))

    fig.add_trace(go.Scatterpolar(
        r=brain_scores,
        theta=expt_names,
        name='brain',
        line={'width': 5, 'color': 'green' },
    
    ))
    
    fig.update_layout(
    title=dict(
        text="Experiment scores for Supervised CNNs and Vision Transformers",
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
    
    fig.show()
    fig.write_image(f'{benchmark_results_path}/arch_radar.png')
        

def main():
    model_scores, (expt_names, expt_y_labels) = load_model_scores()
    brain_scores = load_brain_score()
    model_plot_radar(brain_scores, model_scores, expt_names)

if __name__ == "__main__":
    main()