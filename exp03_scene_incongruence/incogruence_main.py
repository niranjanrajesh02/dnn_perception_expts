import sys
sys.path.insert(1, 'D:/Niranjan_Work/dnns_qualitative/dnn_perception_expts')
import numpy as np
import pickle
from tqdm import tqdm
from _utils.data import load_stim_file, display_np_image
from _utils.network import load_model, get_model_output, get_accuracy
import matplotlib.pyplot as plt
import torch

def get_incongruence_scores(model_name, save=False):
    stim_data_munneke = load_stim_file('./data/munneke_stim.mat') # grayscale 46 total = ((cong, incong) repeated 23 times)
    stim_data_davenport = load_stim_file('./data/davenport_stim.mat') # color 34 = ((cong, incong) repeated 17 times)

    # rearranging so that congruents (x23 or x17) followed by incongruents (x23 or x17)
    stim_data_munneke = np.concatenate((stim_data_munneke[::2], stim_data_munneke[1::2]), axis=0)
    stim_data_davenport = np.concatenate((stim_data_davenport[::2], stim_data_davenport[1::2]), axis=0)
    stim_data = np.concatenate((stim_data_munneke, stim_data_davenport), axis=0)
    # 23 cong_m, 23 incong_m, 17 cong_d, 17 incong_d 


    # manually mapped labels
    labels_munneke = [597, 521, 521, 436, 560, 672, 737, 463, 533, 533, 873, 563, 575, 620, 424, 442, 704, 704, 897, 832, 850, 858, 413, 
                    597, 521, 521, 436, 560, 672, 737, 463, 533, 533, 873, 563, 575, 620, 424, 442, 704, 704, 897, 832, 850, 858, 413]
    labels_davenport = [881, 484, 342, 752, 867, 408, 355, 99, 451, 813, 34, 355, 437, 353, 348, 346, 341, 
                        881, 484, 342, 752, 867, 408, 355, 99, 451, 813, 34, 355, 437, 353, 348, 346, 341]

    labels = np.concatenate((labels_munneke, labels_davenport), axis=0)
    # decrement each label by 1 (because matlab indexing starts from 1)
    labels = labels - 1

    load_model(model_name, with_hooks=False)
    out = get_model_output(stim_data)

    top5_preds = torch.topk(out, 5, dim=1).indices.cpu().numpy()
    top1_preds = torch.topk(out, 1, dim=1).indices.cpu().numpy()

    # congruent preds
    congruent_top5_preds = np.concatenate((top5_preds[0:23], top5_preds[46:63]), axis=0)
    congruent_top1_preds = np.concatenate((top1_preds[0:23], top1_preds[46:63]), axis=0)

    # incongruent preds
    incongruent_top5_preds = np.concatenate((top5_preds[23:46], top5_preds[63:]), axis=0)
    incongruent_top1_preds = np.concatenate((top1_preds[23:46], top1_preds[63:]), axis=0)

    # congruent labels
    congruent_labels = np.concatenate((labels[0:23], labels[46:63]), axis=0)
    # incongruent labels
    incongruent_labels = np.concatenate((labels[23:46], labels[63:]), axis=0)

    # congruent metrics
    ctop5_acc_mean, ctop5_acc_se = get_accuracy(congruent_top5_preds, congruent_labels, topk=5)
    ctop1_acc, ctop1_acc_se = get_accuracy(congruent_top1_preds, congruent_labels, topk=1)


    # incongruent metrics
    itop5_acc_mean, itop5_acc_se = get_accuracy(incongruent_top5_preds, incongruent_labels, topk=5)
    itop1_acc, itop1_acc_se = get_accuracy(incongruent_top1_preds, incongruent_labels, topk=1)

    classification_metrics = {
        'c_top5_acc_mean': ctop5_acc_mean,
        'c_top5_acc_se': ctop5_acc_se,
        'c_top1_acc_mean': ctop1_acc,
        'c_top1_acc_se': ctop1_acc_se,
        'i_top5_acc_mean': itop5_acc_mean,
        'i_top5_acc_se': itop5_acc_se,
        'i_top1_acc_mean': itop1_acc,
        'i_top1_acc_se': itop1_acc_se
    }
    
    # save incongruence metrics
    if save:
        with open(f'./results/{model_name}.pkl', 'wb') as f:
            pickle.dump(classification_metrics, f)
        
        print("Saved Classification Metrics!")
    
    return classification_metrics


if __name__ == "__main__":
    for model in ['vgg16', 'vit_base']:
        get_incongruence_scores(model)
