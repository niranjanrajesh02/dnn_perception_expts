import math
import numpy as np
import matplotlib.pyplot as plt

import re

import timm
import torch
import torchvision
import torch.nn as nn
from torch.nn import Parameter
from torch.nn import functional as F

from functools import partial, reduce
from operator import mul

from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.layers import PatchEmbed

import warnings
warnings.filterwarnings("ignore")

from einops import rearrange, repeat

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
    assert model_name in ["vgg16", "vit_base", "vgg19", "resnet50", "resnet101", "inception_v3", "inception_v4", "convnext_base", "convnext_large", #supervised CNNs
                          "vit_base", "vit_large", "swin_base", "swin_large", "deit_base", "deit_large",                                            #supervised ViTs
                          "resnet50_at", "vit_base_at", "resnet50_moco", "vit_base_moco", "resnet50_dino", "vit_base_dino",                         #training variations
                          "facenet_casia", "facenet_vggface2", "face_vit",                                                                          #face models
                          "resnet50_places365", "densenet161_places365", "vit_base_places365" ]
    global model
    global layers
    layers = []

    #* Supervised CNNs
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
    elif model_name == "densenet161":
        model = timm.create_model('densenet161.tv_in1k', pretrained=True).eval()

    #* Supervised ViTs
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
    
    #* AT 
    elif model_name == "resnet50_at":
        model = timm.create_model('resnet50', pretrained=False)
        ckpt = torch.load('../models/ARES_ResNet50_AT.pth', map_location='cpu')
        model.load_state_dict(ckpt)
        model.eval()

    elif model_name == "vit_base_at":
        model = timm.create_model('vit_base_patch16_224', pretrained=False)
        ckpt = torch.load('../models/ARES_ViT_base_patch16_224_AT.pth', map_location='cpu')
        model.load_state_dict(ckpt)
        model.eval()
    
    #* MOCOv3 SSL
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
        model.eval()

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
        model.eval()
    
    #* DINOv2 SSL
    elif model_name == "resnet50_dino":
        model = torchvision.models.__dict__['resnet50']()
        state_dict = torch.load('../models/resnet50_dino.pth', map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
        model.eval()

    elif model_name == "vit_base_dino":
        model = vit_base()
        state_dict = torch.load('../models/vit_base_patch16_dino.pth', map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
        model.eval()

    #* Face pretrained
    elif model_name == "facenet_casia":
        model = InceptionResnetV1()
        state_dict = torch.load('../models/facenet_casia.pt', map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
        model.eval()
    elif model_name == "facenet_vggface2":
        model = InceptionResnetV1()
        state_dict = torch.load('../models/facenet_vggface2.pt', map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
        model.eval()

    elif model_name == "face_vit":
        gpu_id = torch.device("cuda:0") if torch.cuda.is_available() else None
        model = ViT_face(loss_type="CosFace", GPU_ID=gpu_id, num_class=93431, image_size=112, patch_size=8, dim=512, depth=20, heads=8, mlp_dim=2048, 
                        dropout=0.1,emb_dropout=0.1)
        state_dict = torch.load('../models/face_vit_p8_s8.pth', map_location='cpu')
        # print(model)
        model.load_state_dict(state_dict, strict=False)
        model.eval()

    
    #* Scenes pretrained
    elif model_name == "resnet50_places365":
        model = torchvision.models.__dict__['resnet50'](num_classes=365)
        ckpt = torch.load('../models/resnet50_places365.pth.tar', map_location='cpu')
        state_dict = {str.replace(k,'module.',''): v for k,v in ckpt['state_dict'].items()}
        model.load_state_dict(state_dict)
        model.eval()

   
    elif model_name == "densenet161_places365":
        model = torchvision.models.__dict__['densenet161'](num_classes=365)
        ckpt = torch.load('../models/densenet161_places365.pth.tar', map_location='cpu')
        state_dict = {str.replace(k,'module.',''): v for k,v in ckpt['state_dict'].items()}
        state_dict = {re.sub(r'conv\.(\d+)', r'conv\1', k): v for k, v in state_dict.items()}
        state_dict = {re.sub(r'norm\.(\d+)', r'norm\1', k): v for k, v in state_dict.items()} #old pytorch layer nomenclature had 'conv.1' and 'norm.1' which changed to 'conv1' 'norm1'
        model.load_state_dict(state_dict)
        model.eval()

    elif model_name == "vit_base_places365": #! This model is only fine tuned on Places365
       
        model = torch.load('../models/vit_base_ft_places365.pt', map_location='cpu')
       
        model.eval()
        
    

    
    if torch.cuda.is_available():
        print("CUDA is available. Using GPU")
        model.cuda()
    

    if with_hooks:

        register_hooks(model)

        # sample input into model to check which layers are getting activated
        if "face_vit" in model_name:
            img = torch.randn(1, 3, 112, 112).cuda()
        else:
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


class BasicConv2d(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(
            in_planes, out_planes,
            kernel_size=kernel_size, stride=stride,
            padding=padding, bias=False
        ) # verify bias false
        self.bn = nn.BatchNorm2d(
            out_planes,
            eps=0.001, # value found in tensorflow
            momentum=0.1, # default pytorch value
            affine=True
        )
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Block35(nn.Module):

    def __init__(self, scale=1.0):
        super().__init__()

        self.scale = scale

        self.branch0 = BasicConv2d(256, 32, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(256, 32, kernel_size=1, stride=1),
            BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(256, 32, kernel_size=1, stride=1),
            BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1),
            BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)
        )

        self.conv2d = nn.Conv2d(96, 256, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = self.relu(out)
        return out


class Block17(nn.Module):

    def __init__(self, scale=1.0):
        super().__init__()

        self.scale = scale

        self.branch0 = BasicConv2d(896, 128, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(896, 128, kernel_size=1, stride=1),
            BasicConv2d(128, 128, kernel_size=(1,7), stride=1, padding=(0,3)),
            BasicConv2d(128, 128, kernel_size=(7,1), stride=1, padding=(3,0))
        )

        self.conv2d = nn.Conv2d(256, 896, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = self.relu(out)
        return out


class Block8(nn.Module):

    def __init__(self, scale=1.0, noReLU=False):
        super().__init__()

        self.scale = scale
        self.noReLU = noReLU

        self.branch0 = BasicConv2d(1792, 192, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(1792, 192, kernel_size=1, stride=1),
            BasicConv2d(192, 192, kernel_size=(1,3), stride=1, padding=(0,1)),
            BasicConv2d(192, 192, kernel_size=(3,1), stride=1, padding=(1,0))
        )

        self.conv2d = nn.Conv2d(384, 1792, kernel_size=1, stride=1)
        if not self.noReLU:
            self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        if not self.noReLU:
            out = self.relu(out)
        return out


class Mixed_6a(nn.Module):

    def __init__(self):
        super().__init__()

        self.branch0 = BasicConv2d(256, 384, kernel_size=3, stride=2)

        self.branch1 = nn.Sequential(
            BasicConv2d(256, 192, kernel_size=1, stride=1),
            BasicConv2d(192, 192, kernel_size=3, stride=1, padding=1),
            BasicConv2d(192, 256, kernel_size=3, stride=2)
        )

        self.branch2 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out


class Mixed_7a(nn.Module):

    def __init__(self):
        super().__init__()

        self.branch0 = nn.Sequential(
            BasicConv2d(896, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 384, kernel_size=3, stride=2)
        )

        self.branch1 = nn.Sequential(
            BasicConv2d(896, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 256, kernel_size=3, stride=2)
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(896, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 256, kernel_size=3, stride=1, padding=1),
            BasicConv2d(256, 256, kernel_size=3, stride=2)
        )

        self.branch3 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out

# https://github.com/timesler/facenet-pytorch/blob/master/models/inception_resnet_v1.py
class InceptionResnetV1(nn.Module):
    """Inception Resnet V1 model with optional loading of pretrained weights.

    Model parameters can be loaded based on pretraining on the VGGFace2 or CASIA-Webface
    datasets. Pretrained state_dicts are automatically downloaded on model instantiation if
    requested and cached in the torch cache. Subsequent instantiations use the cache rather than
    redownloading.

    Keyword Arguments:
        pretrained {str} -- Optional pretraining dataset. Either 'vggface2' or 'casia-webface'.
            (default: {None})
        classify {bool} -- Whether the model should output classification probabilities or feature
            embeddings. (default: {False})
        num_classes {int} -- Number of output classes. If 'pretrained' is set and num_classes not
            equal to that used for the pretrained model, the final linear layer will be randomly
            initialized. (default: {None})
        dropout_prob {float} -- Dropout probability. (default: {0.6})
    """
    def __init__(self, pretrained=None, classify=False, num_classes=None, dropout_prob=0.6, device=None):
        super().__init__()

        # Set simple attributes
        self.pretrained = pretrained
        self.classify = classify
        self.num_classes = num_classes

        if pretrained == 'vggface2':
            tmp_classes = 8631
        elif pretrained == 'casia-webface':
            tmp_classes = 10575
        elif pretrained is None and self.classify and self.num_classes is None:
            raise Exception('If "pretrained" is not specified and "classify" is True, "num_classes" must be specified')


        # Define layers
        self.conv2d_1a = BasicConv2d(3, 32, kernel_size=3, stride=2)
        self.conv2d_2a = BasicConv2d(32, 32, kernel_size=3, stride=1)
        self.conv2d_2b = BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.maxpool_3a = nn.MaxPool2d(3, stride=2)
        self.conv2d_3b = BasicConv2d(64, 80, kernel_size=1, stride=1)
        self.conv2d_4a = BasicConv2d(80, 192, kernel_size=3, stride=1)
        self.conv2d_4b = BasicConv2d(192, 256, kernel_size=3, stride=2)
        self.repeat_1 = nn.Sequential(
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
        )
        self.mixed_6a = Mixed_6a()
        self.repeat_2 = nn.Sequential(
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
        )
        self.mixed_7a = Mixed_7a()
        self.repeat_3 = nn.Sequential(
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
        )
        self.block8 = Block8(noReLU=True)
        self.avgpool_1a = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout_prob)
        self.last_linear = nn.Linear(1792, 512, bias=False)
        self.last_bn = nn.BatchNorm1d(512, eps=0.001, momentum=0.1, affine=True)

        if pretrained is not None:
            self.logits = nn.Linear(512, tmp_classes)
            # load_weights(self, pretrained)

        if self.classify and self.num_classes is not None:
            self.logits = nn.Linear(512, self.num_classes)

        self.device = torch.device('cpu')
        if device is not None:
            self.device = device
            self.to(device)

    def forward(self, x):
        """Calculate embeddings or logits given a batch of input image tensors.

        Arguments:
            x {torch.tensor} -- Batch of image tensors representing faces.

        Returns:
            torch.tensor -- Batch of embedding vectors or multinomial logits.
        """
        x = self.conv2d_1a(x)
        x = self.conv2d_2a(x)
        x = self.conv2d_2b(x)
        x = self.maxpool_3a(x)
        x = self.conv2d_3b(x)
        x = self.conv2d_4a(x)
        x = self.conv2d_4b(x)
        x = self.repeat_1(x)
        x = self.mixed_6a(x)
        x = self.repeat_2(x)
        x = self.mixed_7a(x)
        x = self.repeat_3(x)
        x = self.block8(x)
        x = self.avgpool_1a(x)
        x = self.dropout(x)
        x = self.last_linear(x.view(x.shape[0], -1))
        x = self.last_bn(x)
        if self.classify:
            x = self.logits(x)
        else:
            x = F.normalize(x, p=2, dim=1)
        return x
    

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask = None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max
        #embed()
        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)

        return out

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)))
            ]))
    def forward(self, x, mask = None):
        for attn, ff in self.layers:
            x = attn(x, mask = mask)
            #embed()
            x = ff(x)
        return x

class CosFace(nn.Module):
    r"""Implement of CosFace (https://arxiv.org/pdf/1801.09414.pdf):
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        device_id: the ID of GPU where the model will be trained by model parallel.
                       if device_id=None, it will be trained on CPU without model parallel.
        s: norm of input feature
        m: margin
        cos(theta)-m
    """

    def __init__(self, in_features, out_features, device_id, s=64.0, m=0.35):
        super(CosFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device_id = device_id
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------

        if self.device_id == None:
            cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        else:
            x = input
            sub_weights = torch.chunk(self.weight, len(self.device_id), dim=0)
            temp_x = x.cuda(self.device_id[0])
            weight = sub_weights[0].cuda(self.device_id[0])
            cosine = F.linear(F.normalize(temp_x), F.normalize(weight))
            for i in range(1, len(self.device_id)):
                temp_x = x.cuda(self.device_id[i])
                weight = sub_weights[i].cuda(self.device_id[i])
                cosine = torch.cat((cosine, F.linear(F.normalize(temp_x), F.normalize(weight)).cuda(self.device_id[0])),
                                   dim=1)
        phi = cosine - self.m
        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros(cosine.size())
        if self.device_id != None:
            one_hot = one_hot.cuda(self.device_id[0])
        # one_hot = one_hot.cuda() if cosine.is_cuda else one_hot

        one_hot.scatter_(1, label.cuda(self.device_id[0]).view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + (
                    (1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s

        return output

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features = ' + str(self.in_features) \
               + ', out_features = ' + str(self.out_features) \
               + ', s = ' + str(self.s) \
               + ', m = ' + str(self.m) + ')'
class ViT_face(nn.Module):
    def __init__(self, *, loss_type, GPU_ID, num_class, image_size, patch_size, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2
        assert num_patches > 16, f'your number of patches ({num_patches}) is way too small for attention to be effective (at least 16). Try decreasing your patch size'
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.patch_size = patch_size

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
        )
        self.loss_type = loss_type
        self.GPU_ID = GPU_ID
        if self.loss_type == 'None':
            print("no loss for vit_face")
        else:
            if self.loss_type == 'CosFace':
                self.loss = CosFace(in_features=dim, out_features=num_class, device_id=self.GPU_ID)


    def forward(self, img, label= None , mask = None):
        p = self.patch_size

        x = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = p, p2 = p)
        x = self.patch_to_embedding(x)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        x = self.transformer(x, mask)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        emb = self.mlp_head(x)
        if label is not None:
            x = self.loss(emb, label)
            return x, emb
        else:
            return emb  