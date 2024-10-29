import numpy as np
import pandas as pd
import sys
sys.path.insert(1, 'D:/Niranjan_Work/dnns_qualitative/dnn_perception_expts')
from _utils.network import get_model_feature_layer
import ast
import matplotlib.pyplot as plt
import seaborn as sns
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


def expt_bar_plot(brain_scores, model_scores, expt_names):
    expt_titles, expt_y_labels = expt_names

    for expt_i in range(total_expts):
        brain_score = brain_scores[expt_i]

        cnns_scores = {}
        vits_scores = {}
        for model in cnn_list:
            cnns_scores[model] = model_scores[model][expt_i]
        for model in vit_list:
            vits_scores[model] = model_scores[model][expt_i]

        all_scores = np.array(list(cnns_scores.values()) + list(vits_scores.values()))
     
        # barplot of all scores and x axis labels of model names
        plt.figure(figsize=(10, 8))

        model_names = list(cnns_scores.keys()) + list(vits_scores.keys()) 
        colours = ['blue'] * len(cnns_scores) + ['orange'] * len(vits_scores)
        sns.barplot(x=model_names, y=all_scores, palette=colours)
        # x ticks 45 deg
        plt.xticks(rotation=45)
        plt.title(expt_titles[expt_i])
        plt.axhline(y=brain_score, color='green', linestyle='--')
        plt.ylim(-1,1)
        if expt_i ==2:
            plt.ylabel('Score')
        else:
            plt.ylabel(expt_y_labels[expt_i])


        plot_path = '../plots/scores'
        expt_code = expt_titles[expt_i].split(':')[0]
        plt.savefig(f'{plot_path}/{expt_code}.png')
    
       

        
    

def main():
    model_scores, expt_names = load_model_scores()   
    brain_scores = load_brain_score()
    expt_bar_plot(brain_scores, model_scores, expt_names)

if __name__ == "__main__":
    main()