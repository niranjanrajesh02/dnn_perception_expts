import numpy as np
import torch
from PIL import Image
import timm
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
import shelve
import mmap
import struct

model = None

# function to get activations from a layer
activations = {}
# activations = []
def get_activation(name):
    def hook(model, input, output):
        activations[name] = np.squeeze(output.cpu().detach()) #squeeze-removes batchdim, cpu-moves to cpu, detach-removes gradient
    return hook



# loads model and registers hooks
def load_model(model_name, with_hooks=True):
    assert model_name in ["vgg16", "vit_base"]

    global model
    if model_name == "vgg16":
        model = timm.create_model('vgg16_bn.tv_in1k', pretrained=True).eval()
    elif model_name == "vit_base":
        model = timm.create_model('vit_base_patch16_224', pretrained=True).eval()
    
    if torch.cuda.is_available():
        model.cuda()
    
    layers = []

    if with_hooks:
        for layer, child in model.named_children():
            if len(list(child.children())) > 0:
                    for layer_c, child_c in child.named_children():
                            layer_name = f'{layer}_{layer_c}'
                            if len(list(child_c.children())) > 0:
                                for layer_cc, child_cc in child_c.named_children():
                                    layer_name = f'{layer}_{layer_c}_{layer_cc}'
                                    child_cc.register_forward_hook(get_activation(layer_name))
                                    layers.append(layer_name)
                            else:
                                child_c.register_forward_hook(get_activation(layer_name))
                                layers.append(layer_name)
            else:
                layer_name = layer
                child.register_forward_hook(get_activation(layer_name))
                layers.append(layer_name)
            print(f"Loaded {model_name} with {len(layers)} layers")
            return layers
    else:
        print("Model loaded")
        return
    
# for an image, get activations from each hooked layer
def get_layerwise_activations(img):
    global model
    # extract activations
    img_representations = []
    img = torch.from_numpy(img)
    if torch.cuda.is_available():
        img = img.cuda()
    with torch.no_grad():
        # add batch dim
        img = img.unsqueeze(0)
        out = model(img)
        global activations
        img_representations.append(activations)
        activations = {} # reset for next img

    return img_representations[0]

# for a set of image, get their model output
def get_model_output(imgs):
    global model
    imgs = torch.from_numpy(imgs)
    if torch.cuda.is_available():
        imgs = imgs.cuda()
    with torch.no_grad():
        out = model(imgs.float())

    return out


# main function
def get_representations(net, imgs, save=False):
    
    reps = extract_layerwise_features(net, imgs)


# accuracy calculation

def get_accuracy(preds, labels, topk=1):
    assert len(preds) == len(labels)
    correct = 0
    if topk == 1:
        for pred, label in zip(preds, labels):
            if pred == label:
                correct += 1
    elif topk > 1:
        for pred, label in zip(preds, labels):
            if label in pred:
                correct += 1
    
    avg_accuracy = correct / len(labels)
    se_accuracy = np.sqrt(avg_accuracy * (1 - avg_accuracy) / len(labels))

    return avg_accuracy, se_accuracy