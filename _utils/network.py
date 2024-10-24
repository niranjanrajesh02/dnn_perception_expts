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
import torch.nn as nn
import torchvision
model = None

# function to get activations from a layer
activations = {}
# activations = []
def get_activation(name):
    def hook(model, input, output):

        activations[name] = np.squeeze(output.cpu().detach()) #squeeze-removes batchdim, cpu-moves to cpu, detach-removes gradient
    return hook

layers = []
def register_hooks(module, parent_name=""):
    # Iterate over the module's children (submodules)
    for name, child in module.named_children():
        # Create the full name for the layer
        full_name = f'{parent_name}_{name}' if parent_name else name
        
        if isinstance(child, (nn.Dropout)):
            # print(f"Skipping layer: {full_name} (Dropout)")
            continue
        # If the child module has its own submodules, recurse
        if len(list(child.children())) > 0:
            register_hooks(child, full_name)  
        else:
            child.register_forward_hook(get_activation(full_name))
            layers.append(full_name)


# loads model and registers hooks
def load_model(model_name, with_hooks=True):
    assert model_name in ["vgg16", "vit_base", "vgg19", "resnet50", "resnet101", "inception_v3", "inception_v4", "convnext_base", "convnext_large",
                          "vit_base", "vit_large", "swin_base", "swin_large", "deit_base", "deit_large"]

    global model
    global layers
    layers = []
    # Supervised CNNs
    if model_name == "vgg16":
        model = timm.create_model('vgg16_bn.tv_in1k', pretrained=True).eval()
    elif model_name == "vgg19":
        model = timm.create_model('vgg19_bn.tv_in1k', pretrained=True).eval()
    elif model_name == "resnet50":
        model = timm.create_model('resnet50.tv2_in1k', pretrained=True).eval()
    elif model_name == "resnet101":
        model = timm.create_model('resnet101.tv2_in1k', pretrained=True).eval()
    elif model_name == "inception_v3":
        model = timm.create_model('inception_v3.tv_in1k', pretrained=True).eval()
    elif model_name == "inception_v4":
        model = timm.create_model('inception_v4.tf_in1k', pretrained=True).eval()
    elif model_name == "convnext_base":
        model = timm.create_model('convnext_base.fb_in1k', pretrained=True).eval()
    elif model_name == "convnext_large":
        model = timm.create_model('convnext_large.fb_in1k', pretrained=True).eval()

    #  Supervised ViTs
    elif model_name == "vit_base":
        model = torchvision.models.vit_b_16(weights="DEFAULT").eval()
        # model = timm.create_model('vit_base_patch16_224.orig_in21k_ft_in1k', pretrained=True).eval()
    elif model_name == "vit_large":
        model = torchvision.models.vit_l_16(weights="DEFAULT").eval()
    elif model_name == "swin_base":
        model = timm.create_model('swin_large_patch4_window7_224.ms_in1k', pretrained=True).eval()
    elif model_name == "swin_large":
        model = timm.create_model('swin_large_patch4_window7_224.ms_in22k_ft_in1k', pretrained=True).eval() #! not in1k
    elif model_name =="deit_base":
        model = timm.create_model('deit3_base_patch16_224.fb_in1k', pretrained=True).eval()
    elif model_name == "deit_large":
        model = timm.create_model('deit3_large_patch16_224.fb_in1k', pretrained=True).eval()
    
    
    if torch.cuda.is_available():
        model.cuda()
    

    if with_hooks:

        register_hooks(model)

        # sample input into model to check which layers are getting activated
        img = torch.randn(1, 3, 224, 224).cuda()
        out = model(img)
        global activations
        # check if activations keys match with layers
        if len(activations) != len(layers):
            # remove layers if not present in activations keys
            layers = [layer for layer in layers if layer in activations.keys()]
            
        activations = {}
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