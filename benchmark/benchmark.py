import sys
sys.path.insert(1, 'D:/Niranjan_Work/dnns_qualitative/dnn_perception_expts')
import os
import pandas as pd
import numpy as np
import shutil
from _utils.network import load_model
from exp01_thatcher_effect.thatcher_main import get_thatcher_scores
from exp02_mirror_confusion.mirror_main import get_mirror_scores
from exp03_scene_incongruence.incogruence_main import get_incongruence_scores
from exp04_divisive_normalization.div_norm_main import get_div_norm_scores
from exp05_correlated_sparseness.corr_sparse_main import get_corr_sparse_scores
from exp06_webers_law.webers_main import get_weber_law_correlation
from exp07_relative_size.rel_size_main import get_relative_size_index
from exp08_surface_invariance.surf_main import get_surface_invariance_index
from exp09_3d.inv_3d_main import get_3d_scores
from exp10_occlusion.occlusion_main import get_occlusion_scores
from exp11_object_parts.obj_parts_main import get_obj_parts_scores
from exp12_global_local.global_main import get_global_advantage
import argparse

experiments = {
    "exp01_thatcher_effect": get_thatcher_scores,
    "exp02_mirror_confusion": get_mirror_scores,
    "exp03_scene_incongruence": get_incongruence_scores,
    "exp04_divisive_normalization": get_div_norm_scores,
    "exp05_correlated_sparseness": get_corr_sparse_scores,
    "exp06_webers_law": get_weber_law_correlation,
    "exp07_relative_size": get_relative_size_index,
    "exp08_surface_invariance": get_surface_invariance_index,
    "exp09_3d": get_3d_scores,
    "exp10_occlusion": get_occlusion_scores,
    "exp11_object_parts": get_obj_parts_scores,
    "exp12_global_local": get_global_advantage
}

# cli args
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='vit_base')
args = parser.parse_args()


models_to_run = [args.model]
results_path = './benchmark_results'

for model_name in models_to_run:
    layers = load_model(model_name)
    model_results_path = f'{results_path}/{model_name}'

    if os.path.exists(model_results_path):
        
        print(f"Deleting {model_results_path}")
        shutil.rmtree(model_results_path)
        os.mkdir(model_results_path)

    else:
        os.mkdir(model_results_path)

    model_score_df = pd.DataFrame(index=np.arange(1,len(experiments)+1), columns=layers)
    model_err_df = pd.DataFrame(index=np.arange(1,len(experiments)+1), columns=layers)

    for exp_i, exp in enumerate(experiments):
        exp_name = list(experiments.keys())[exp_i]
        print(f"\n\n=================EXPT {exp_i+1}: {exp_name}==================\n") #df index start from 1, not 0!
        os.chdir(f'../{exp_name}')
        if exp_name == "exp03_scene_incongruence":
            model_score_df.loc[exp_i+1] = [np.nan for _ in range(len(layers))]
            exp_result = experiments[exp](model_name)
            model_score_df.loc[exp_i+1, layers[0]] = [exp_result]
            model_err_df.loc[exp_i+1] = [np.nan for _ in range(len(layers))]
        
        else:
            exp_scores, exp_err = experiments[exp](model_name)
            if len(exp_err) == 0:
                exp_err = [np.nan for _ in range(len(layers))]
            # print(exp_scores.shape) #* for debug
            
            # print(exp_scores) #* for debug

            # if each layer has more than one score (multiple conditions), store as list
            if len(exp_scores.shape) > 1:
                model_score_df.loc[exp_i+1] = [[] for _ in range(len(layers))]
                model_err_df.loc[exp_i+1] = [[] for _ in range(len(layers))]
                for score in exp_scores:
                    for col, value in zip(model_score_df.columns, score):  
                        model_score_df.loc[exp_i+1, col].append(value)
           
                for err in exp_err:
                    if len(err) == 0:
                        err = [np.nan for _ in range(len(layers))]
                    for col, value in zip(model_err_df.columns, err):
                        model_err_df.loc[exp_i+1, col] = [np.nan for _ in range(len(layers))]
                        model_err_df.loc[exp_i+1, col].append(value)
            else:
    
                model_score_df.loc[exp_i+1] = exp_scores
                model_err_df.loc[exp_i+1] = exp_err
        os.chdir('../benchmark')

    model_score_df.to_csv(f'{model_results_path}/{model_name}_scores.csv')
    model_err_df.to_csv(f'{model_results_path}/{model_name}_sem.csv')


