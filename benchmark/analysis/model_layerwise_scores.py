import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

cnn_list = ['vgg16', 'vgg19', 'resnet50', 'resnet101', 'inception_v3', 'inception_v4', 'convnext_base', 'convnext_large']
vit_list = ['vit_base', 'vit_large', 'swin_base', 'swin_large', 'deit_base', 'deit_large']

plot_models = {'CNNs': cnn_list, 'ViTs': vit_list}


benchmark_results_path = '../benchmark_results'
plot_save_path = '../plots/layerwise' 
# 1 expt, all models
def plot_model_layerwise_scores(expt):
    # get 'brain score"
    brain_df = pd.read_csv(f'{benchmark_results_path}/brain/brain_proc_scores.csv', index_col=0)
    brain_score = dict(brain_df.iloc[expt-1])
    

   
    for model_family, model_list in plot_models.items():
        
        plt.figure(figsize=(10, 6))
        plt.axhspan(0, plt.gca().get_ylim()[1], color='gray', alpha=0.3) # Human perception zone

    
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
            
            if num_layers <= 100:
                layer_indices = np.linspace(0, 100, num_layers) # x values: 0-100, num_layers points
                normalised_layer_scores = list(model_layerwise_score.values())
            else:
                window_size = int(np.floor(num_layers / 100))
                # for every score in window_size, append the average
                counter = 0
                for i in range(0, num_layers, window_size):
                    if counter <100:
                        window = list(model_layerwise_score.values())[i:i+window_size]
                    else: #ensure num_windows = 100
                        window = list(model_layerwise_score.values())[i:] # 100th window is the remaining values
                        break
                    window_avg = np.mean(window)
                    normalised_layer_scores.append(window_avg)
                    counter += 1
                layer_indices = np.arange(len(normalised_layer_scores))
            sns.lineplot(x=layer_indices, y=normalised_layer_scores, label=model)
    
        expt_title = dict(model_scores_df.iloc[expt-1])['expt_name']
        expt_y_label = dict(model_scores_df.iloc[expt-1])['expt_y_label']    

        
        ref_label = brain_score['reference']
        ref_val = float(brain_score['scores'])
        label_pos = 0.01
        if expt == 4 or expt ==5:
            label_pos = -0.1
        plt.axhline(y=ref_val, xmin=0, xmax=100, color='black', linestyle='--')
        plt.text(5, ref_val+label_pos, ref_label, fontsize=12, color='black', ha='left', va='bottom')
        plt.title(f'{expt_title} - {model_family}')
        plt.xlabel('Model Depth (%)')
        plt.ylabel(expt_y_label)
        plt.ylim(-1, 1)
        plt.xlim(0,100)
        plt.legend(title="Network")
        expt_code = expt_title.split(':')[0]
        plt.savefig(f'{plot_save_path}/{expt_code}_{model_family}.png')
        plt.close()
        

for expt in range(1,18):
    if expt != 3:
        plot_model_layerwise_scores(expt)