import numpy as np
import torch
import timm
import matplotlib.pyplot as plt
import torch.nn as nn
import torchvision
import math

from functools import partial, reduce
from operator import mul

from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.layers import PatchEmbed

import warnings
warnings.filterwarnings("ignore")

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
                          "vit_base", "vit_large", "swin_base", "swin_large", "deit_base", "deit_large",
                          "resnet50_at", "vit_base_at", "resnet50_moco", "vit_base_moco", "resnet50_dino", "vit_base_dino"]

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
        model = timm.create_model('swin_base_patch4_window7_224.ms_in1k', pretrained=True).eval()
    elif model_name == "swin_large":
        model = timm.create_model('swin_large_patch4_window7_224.ms_in22k_ft_in1k', pretrained=True).eval() #! not in1k
    elif model_name =="deit_base":
        model = timm.create_model('deit3_base_patch16_224.fb_in1k', pretrained=True).eval()
    elif model_name == "deit_large":
        model = timm.create_model('deit3_large_patch16_224.fb_in1k', pretrained=True).eval()
    
    # AT 
    elif model_name == "resnet50_at":
        model = timm.create_model('resnet50', pretrained=False)
        ckpt = torch.load('../models/ARES_ResNet50_AT.pth', map_location='cpu')
        model.load_state_dict(ckpt)
    elif model_name == "vit_base_at":
        model = timm.create_model('vit_base_patch16_224', pretrained=False)
        ckpt = torch.load('../models/ARES_ViT_base_patch16_224_AT.pth', map_location='cpu')
        model.load_state_dict(ckpt)
    
    # MOCOv3 SSL
    elif model_name == "resnet50_moco":
        linear_keyword = 'fc'
        state_dict = torch.load('../models/resnet50_mocov3.tar', map_location='cpu')['state_dict']
        for k in list(state_dict.keys()):
                # retain only base_encoder up to before the embedding layer
                if k.startswith('module.base_encoder') and not k.startswith('module.base_encoder.%s' % linear_keyword):
                    # remove prefix
                    state_dict[k[len("module.base_encoder."):]] = state_dict[k]
                # delete renamed or unused k
                del state_dict[k]
        model = torchvision.models.__dict__['resnet50']()
        model.load_state_dict(state_dict, strict=False)

    elif model_name == "vit_base_moco":
        linear_keyword = 'head'
        state_dict = torch.load('../models/vit_base_mocov3.tar', map_location='cpu')['state_dict']
        for k in list(state_dict.keys()):
                # retain only base_encoder up to before the embedding layer
                if k.startswith('module.base_encoder') and not k.startswith('module.base_encoder.%s' % linear_keyword):
                    # remove prefix
                    state_dict[k[len("module.base_encoder."):]] = state_dict[k]
                # delete renamed or unused k
                del state_dict[k]
        model = vit_base()
        model.load_state_dict(state_dict, strict=False)
    
    # DINOv2 SSL
    elif model_name == "resnet50_dino":
        model = torchvision.models.__dict__['resnet50']()
        state_dict = torch.load('../models/resnet50_dino.pth', map_location='cpu')
        model.load_state_dict(state_dict, strict=False)

    elif model_name == "vit_base_dino":
        model = vit_base()
        state_dict = torch.load('../models/vit_base_patch16_dino.pth', map_location='cpu')
        model.load_state_dict(state_dict, strict=False)

    

    
    if torch.cuda.is_available():
        print("CUDA is available. Using GPU")
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

class VisionTransformerMoCo(VisionTransformer):
    def __init__(self, stop_grad_conv1=False, **kwargs):
        super().__init__(**kwargs)
        # Use fixed 2D sin-cos position embedding
        self.build_2d_sincos_position_embedding()

        # weight initialization
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                if 'qkv' in name:
                    # treat the weights of Q, K, V separately
                    val = math.sqrt(6. / float(m.weight.shape[0] // 3 + m.weight.shape[1]))
                    nn.init.uniform_(m.weight, -val, val)
                else:
                    nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.normal_(self.cls_token, std=1e-6)

        if isinstance(self.patch_embed, PatchEmbed):
            # xavier_uniform initialization
            val = math.sqrt(6. / float(3 * reduce(mul, self.patch_embed.patch_size, 1) + self.embed_dim))
            nn.init.uniform_(self.patch_embed.proj.weight, -val, val)
            nn.init.zeros_(self.patch_embed.proj.bias)

            if stop_grad_conv1:
                self.patch_embed.proj.weight.requires_grad = False
                self.patch_embed.proj.bias.requires_grad = False

    def build_2d_sincos_position_embedding(self, temperature=10000.):
        h, w = self.patch_embed.grid_size
        grid_w = torch.arange(w, dtype=torch.float32)
        grid_h = torch.arange(h, dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h)
        assert self.embed_dim % 4 == 0, 'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = self.embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature**omega)
        out_w = torch.einsum('m,d->md', [grid_w.flatten(), omega])
        out_h = torch.einsum('m,d->md', [grid_h.flatten(), omega])
        pos_emb = torch.cat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], dim=1)[None, :, :]

        # assert self.num_tokens == 1, 'Assuming one and only one token, [cls]'
        pe_token = torch.zeros([1, 1, self.embed_dim], dtype=torch.float32)
        self.pos_embed = nn.Parameter(torch.cat([pe_token, pos_emb], dim=1))
        self.pos_embed.requires_grad = False
def vit_base(**kwargs):
    model = VisionTransformerMoCo(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model