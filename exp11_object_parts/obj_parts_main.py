import os
import sys
sys.path.insert(1, 'D:/Niranjan_Work/dnns_qualitative/dnn_perception_expts/exp11_object_parts')
from obj_parts_a import get_part_matching_index
from obj_parts_b import get_part_correlations
import numpy as np


def get_obj_parts_scores(model, save=False):
    part_match_index, part_sems = get_part_matching_index(model, save)
    print(os.getcwd())
    part_correlations = get_part_correlations(model, save)
    obj_parts_scores = [part_match_index, part_correlations]
    return np.array(obj_parts_scores), [part_sems,[]]

if __name__ == "__main__":
    for model in ['vgg16', 'vit_base']:
       get_obj_parts_scores(model)

os.chdir(f'../benchmark')