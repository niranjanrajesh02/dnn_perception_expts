import numpy as np
import pandas as pd

benchmark_results_path = './benchmark_results'

def load_model_scores(model):

    model_results_path = f'{benchmark_results_path}/{model}'
    model_scores_df = pd.read_csv(f'{model_results_path}/{model}_scores.csv', index_col=0)  # each row is an experiment
    
    model_benchmark_score = []

    return

if __name__ == "__main__":
    load_model_scores('vgg16')