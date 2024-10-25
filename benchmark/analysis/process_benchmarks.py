import pandas as pd
import os
import numpy as np
import ast

benchmark_results_path = '../benchmark_results'

cnn_list = ['vgg16', 'vgg19', 'resnet50', 'resnet101', 'inception_v3', 'inception_v4', 'convnext_base', 'convnext_large']
vit_list = ['vit_base', 'vit_large', 'swin_base', 'swin_large', 'deit_base', 'deit_large']
expt_names = ["Exp01: Thatcher Index", "Exp02: Mirror Confusion", "Exp03: Scene Incongruence",
            "Exp04a: Multiple Object Normalisation (Pairs)", "Exp04b: Multiple Object Normalisation (Triplets)",
            "Exp05a: Correlated Sparseness (Reference set vs Morphlines)","Exp05b: Correlated Sparseness (Shapes vs Textures)",
            "Exp06: Weber's Law", "Exp07: Relative Size Encoding", "Exp08: Surface Invariance", 
            "Exp09a: 3D Processing (Condition 1)", "Exp09b: 3D Processing (Condition 2)",
            "Exp10a: Occlusion Processing (Basic)", "Exp10b: Occlusion Processing (Depth Ordering)",
            "Exp11a: Object Parts Processing (Part Matching)", "Exp11b: Object Parts Processing (Part Correlations)",
            "Exp12: Global-Local Shape Processing"]
expt_y_labels = ["Thatcher Index", "Mirror Confusion Index", "",
                "Estimated Slope", "Estimated Slope",
                "Sparseness Correlation", "Sparseness Correlation",
                "Correlation with length changes", "Average Relative Size Index", "Average Surface Invariance Index",
                "3D Processing Index", "3D Processing Index",
                "Occlusion Index", "Occlusion Index",
                "Part-matching Index", "Natural Part Advantage",
                "Global Advantage Index"]

double_expts = [4,5,9,10,11]
model_list = cnn_list + vit_list

for model in model_list:
    model_results_path = f'{benchmark_results_path}/{model}'

    if not os.path.exists(model_results_path):
        print("No benchmark results for", model)
        break

    model_scores_df = pd.read_csv(f'{model_results_path}/{model}_scores.csv', index_col=0)  # each row is an experiment
    # print(model_scores_df)
    total_expts = 17 # includes sub-expts !TODO: add expt3
    new_model_scores_df = pd.DataFrame(index=np.arange(1,total_expts+1), columns=model_scores_df.columns)

    new_index = 1
    for exp_i in model_scores_df.index:
        if exp_i not in double_expts and exp_i != 3:
            scores = dict(model_scores_df.loc[exp_i])
            scores_i = {k: float(v) for k, v in scores.items()}
            new_model_scores_df.loc[new_index] = model_scores_df.loc[exp_i]
            new_index += 1
        elif exp_i == 3:
            new_model_scores_df.loc[new_index] = model_scores_df.loc[exp_i]
            new_index += 1
        elif exp_i in double_expts:
            
            scores = dict(model_scores_df.loc[exp_i])
            
            scores = {k: v.replace("nan", "None") for k, v in scores.items()}
            scores = {k: ast.literal_eval(v) for k, v in scores.items()}
            for i in range(2):
                # print(scores)
                # print(exp_i)
                scores_i = {k: v[i] for k, v in scores.items()}
                new_model_scores_df.loc[new_index] = scores_i
                new_index += 1

    # add new columns of expt_name and expt_y_label at the beginning
    new_model_scores_df.insert(0, 'expt_name', expt_names)
    new_model_scores_df.insert(1, 'expt_y_label', expt_y_labels)
    

    new_model_scores_df.to_csv(f'{model_results_path}/{model}_proc_scores.csv')
    

