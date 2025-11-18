import math
import warnings
from tqdm import tqdm

import torch
import random
import numpy as np

from torchvision.models import resnet50, convnext_tiny, vit_b_16
from typing import Dict, Any, Optional, Tuple
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import torch
from tqdm import tqdm

import os
import pandas as pd

import copy

from polychannels import RESNET_POLY_CHANNELS_0_01, RESNET_POLY_CHANNELS_0_02, RESNET_POLY_CHANNELS_0_03, RESNET_MULTI_POLY_CHANNELS_0_03, RESNET_MULTI_POLY_CHANNELS_0_02, RESNET_MULTI_POLY_CHANNELS_0_01, VIT_MULTI_POLY_CHANNELS_0_01, VIT_MULTI_POLY_CHANNELS_0_02, VIT_MULTI_POLY_CHANNELS_0_03, DINOv2

from polychannels2 import DINOv2_0001


"""
Sparse Autoencoder (SAE) Utilities

This module provides utility functions and classes for training and using
Sparse Autoencoders, including dataset handling, learning rate schedulers,
custom activation functions, and various mathematical operations.
"""

def set_layer_by_path(model, path, new_layer):
    """Set a layer in model using dot notation path"""
    parts = path.split('.')
    current = model
    
    # Navigate to parent of target layer
    for part in parts[:-1]:
        if part.isdigit():
            current = current[int(part)]
        else:
            current = getattr(current, part)
    
    # Set the final layer
    final_part = parts[-1]
    if final_part.isdigit():
        current[int(final_part)] = new_layer
    else:
        setattr(current, final_part, new_layer)


def get_layer_by_path(model, path):
    """Navigate to a layer using dot notation path."""
    parts = path.split('.')
    current = model
    
    for part in parts:
        if part.isdigit():
            current = current[int(part)]
        else:
            current = getattr(current, part)
    
    return current

class MultiClassTrainingModel(torch.nn.Module):
    def __init__(self, channel_of_interest, input_channels, kernel_size, stride, num_classes, padding=0, bias=False):
        super(MultiClassTrainingModel, self).__init__()

        self.disentangled = nn.Conv2d(input_channels, num_classes+1, kernel_size=kernel_size, stride=stride, bias=bias, padding=padding)
        self.merged = nn.Conv2d(num_classes+1, 1, kernel_size=1, stride=1, bias=False)
        merged_weight = self.merged.weight.clone()
        for i in range(num_classes):
            merged_weight[0,i,0,0] = 1.
        merged_weight[0,num_classes,0,0] = -1.
        self.merged.weight = nn.Parameter(merged_weight)

        self.channel_of_interest = channel_of_interest

    def forward(self, x):

        disentangled = self.disentangled(x)
        merged = self.merged(disentangled)

        return merged, disentangled


class MultiClassTrainingModelViT(torch.nn.Module):
    def __init__(self, channel_of_interest, input_channels, num_classes, bias=False):
        super(MultiClassTrainingModelViT, self).__init__()

        self.disentangled = nn.Linear(input_channels, num_classes+1, bias=bias)
        self.merged = nn.Linear(num_classes+1, 1, bias=bias)
        merged_weight = self.merged.weight.clone()
        for i in range(num_classes):
            merged_weight[0,i] = 1.
        merged_weight[0,num_classes] = -1.
        self.merged.weight = nn.Parameter(merged_weight)

        self.channel_of_interest = channel_of_interest

    def forward(self, x):

        disentangled = self.disentangled(x)
        merged = self.merged(disentangled)

        return merged, disentangled

def load_multi_disentangled_resnet(model, layer_path, channel_of_interest, classes, device="cuda:0", f=None, model_dir='/data/vimb06/resnet_store_dir/models_disentangled'):
    model = model.to(device)
    print(channel_of_interest, len(classes))
    
    model_disentangled_final = copy.deepcopy(model)
    model_disentangled_final = model_disentangled_final.to(device)
    model_disentangled_final.eval()
    
    model_type = 'resnet50'
    model_disentangled = MultiClassTrainingModel(channel_of_interest, 512, 1, 1, num_classes=len(classes)).to(device)
    model_disentangled = model_disentangled.to(device)

    f = f
    if f == None:
        f = '_best'

    model_disentangled.load_state_dict(torch.load(os.path.join(model_dir, model_type + '_channel' + str(channel_of_interest) + '_f' + str(f) + '_unnormalized.pth')))
    model_layer = get_layer_by_path(model, layer_path)
    iden = None

    if isinstance(model_layer, nn.Sequential):
        iden = model_layer[1]
        model_layer = model_layer[0]
    in_channels, out_channels = model_layer.in_channels, model_layer.out_channels

    model_disentangled_final_layer = get_layer_by_path(model_disentangled_final, layer_path)
    model_disentangled_final_layer = nn.Sequential(
            nn.Conv2d(512, out_channels+len(classes), kernel_size=1, stride=1, bias=False),
            nn.Conv2d(out_channels+len(classes), 2048, kernel_size=1, stride=1, bias=False)
    ).to(device)

    new_weight_Lmin1_L = model_layer.weight.clone() #model.layer4[2].conv3.weight.clone()

    new_weight_Lmin1_L[channel_of_interest, :,:,:] = model_disentangled.disentangled.weight[0,:,:,:].clone()
    for i in range(1, len(classes)+1):
        new_weight_Lmin1_L = torch.cat([new_weight_Lmin1_L,model_disentangled.disentangled.weight[i:i+1,:,:,:].clone()], dim=0) 

    model_disentangled_final_layer[0].weight = nn.Parameter(new_weight_Lmin1_L)

    if not iden is None:
        prev_identity = iden.weight.clone()  # Copy previous identity matrix
        identity = torch.zeros((2048, out_channels+len(classes), 1, 1)).to(device)
        identity[:prev_identity.shape[0], :prev_identity.shape[1], :, :] = prev_identity
    else:
        identity = torch.zeros((2048, out_channels+len(classes), 1, 1)).to(device)
        for i in range(out_channels):
            identity[i,i,0,0] = 1.

    for i in range(len(classes)):
        identity[channel_of_interest,out_channels+i,0,0] = 1. 
    
    identity[channel_of_interest,out_channels+len(classes)-1,0,0] = -1. * (len(classes)-1)
    #model_disentangled_final.layer4[2].conv3[1].weight = nn.Parameter(identity)
    model_disentangled_final_layer[1].weight = nn.Parameter(identity)
    set_layer_by_path(model_disentangled_final, layer_path, model_disentangled_final_layer)
    return model_disentangled_final

def load_fully_multi_disentangled_resnet(tau=0.03):
    model = torchvision.models.resnet50(pretrained=True)
    layer_path = 'layer4.2.conv3'
    channel_to_disentangle = None
    model_dir = '/scratch/inf0/user/dbagci/resnet50/models_disentangled'
    if tau==0.03:
        channel_to_disentangle = RESNET_MULTI_POLY_CHANNELS_0_03
        model_dir = '/scratch/inf0/user/dbagci/resnet50/models_disentangled_0.03'
    elif tau==0.02:
        channel_to_disentangle = RESNET_MULTI_POLY_CHANNELS_0_02
        model_dir = '/scratch/inf0/user/dbagci/resnet50/models_disentangled_0.02'
    elif tau==0.01:
        channel_to_disentangle = RESNET_MULTI_POLY_CHANNELS_0_01
        model_dir = '/scratch/inf0/user/dbagci/resnet50/models_disentangled_0.01'
    else:
        assert False, "Tau not available"

    model2 = copy.deepcopy(model).to("cuda:0")
    random_input = torch.randn((1,3,224,224)).to("cuda:0")
    for entry in channel_to_disentangle:
        set_of_classes = channel_to_disentangle[entry]

        model = load_multi_disentangled_resnet(model, layer_path, int(entry), set_of_classes, model_dir=model_dir).to("cuda:0")
        #print('Are equal:')
        #print(torch.isclose(model(random_input), model2(random_input), atol=1e-03).all())
        #print(model.layer4[2].conv3[0])
        #print(model.layer4[2].conv3[0].weight.shape)
        layer_path = 'layer4.2.conv3' 
    
    return model, 'layer4.2.conv3.0'

def load_multi_disentangled_vit(model, layer_path, channel_of_interest, classes, device="cuda:0", f=None, model_dir='/data/vimb06/resnet_store_dir/models_disentangled'):
    model = model.to(device)
    
    model_disentangled_final = copy.deepcopy(model)
    model_disentangled_final = model_disentangled_final.to(device)
    model_disentangled_final.eval()
    
    model_type = 'vitb16'
    model_disentangled = MultiClassTrainingModelViT(channel_of_interest, 3072, num_classes=len(classes)).to(device)
    model_disentangled = model_disentangled.to(device)

    f = f
    if f == None:
        f = '_best'

    if "0.01" in model_dir:
        model_type = 'vitb16_cls'#model_dir=model_dir.replace("vitb16", "vitb16_cls")
        model_disentangled.load_state_dict(torch.load(os.path.join(model_dir, model_type + '_channel' + str(channel_of_interest) + '_f' + str(f) + '_unnormalized.pth')))
    else:
        model_disentangled.load_state_dict(torch.load(os.path.join(model_dir, model_type + '_channel' + str(channel_of_interest) + '_f' + str(f) + '_unnormalized.pth')))
    model_layer = get_layer_by_path(model, layer_path)
    iden = None

    if isinstance(model_layer, nn.Sequential):
        iden = model_layer[1]
        model_layer = model_layer[0]
    in_channels, out_channels = model_layer.in_features, model_layer.out_features

    model_disentangled_final_layer = get_layer_by_path(model_disentangled_final, layer_path)
    model_disentangled_final_layer = nn.Sequential(
            nn.Linear(3072, out_channels+len(classes), bias=True),
            nn.Linear(out_channels+len(classes), 768, bias=False)
    ).to(device)

    new_weight_Lmin1_L = model_layer.weight.clone() #model.layer4[2].conv3.weight.clone()

    new_weight_Lmin1_L[channel_of_interest, :] = model_disentangled.disentangled.weight[0,:].clone()
    for i in range(1, len(classes)+1):
        new_weight_Lmin1_L = torch.cat([new_weight_Lmin1_L,model_disentangled.disentangled.weight[i:i+1,:].clone()], dim=0) 

    model_disentangled_final_layer[0].weight = nn.Parameter(new_weight_Lmin1_L)
    
    if model_layer.bias is not None:
        new_bias = model_layer.bias.clone()
        # Add bias for the new disentangled neurons (len(classes) new neurons total)
        # All new neurons get the bias from the original channel_of_interest
        for i in range(len(classes)):
            new_bias = torch.cat([new_bias, model_layer.bias[channel_of_interest:channel_of_interest+1].clone()], dim=0)
        model_disentangled_final_layer[0].bias = nn.Parameter(new_bias)
    
    if not iden is None:
        prev_identity = iden.weight.clone()  # Copy previous identity matrix
        identity = torch.zeros((768, out_channels+len(classes))).to(device)
        identity[:prev_identity.shape[0], :prev_identity.shape[1]] = prev_identity
    else:
        identity = torch.zeros((768, out_channels+len(classes))).to(device)
        for i in range(out_channels):
            identity[i,i] = 1.

    for i in range(len(classes)):
        identity[channel_of_interest,out_channels+i] = 1. 
    
    identity[channel_of_interest,out_channels+len(classes)-1] = -1. * (len(classes)-1)
    #model_disentangled_final.layer4[2].conv3[1].weight = nn.Parameter(identity)
    model_disentangled_final_layer[1].weight = nn.Parameter(identity)
    set_layer_by_path(model_disentangled_final, layer_path, model_disentangled_final_layer)
    return model_disentangled_final


def load_fully_multi_disentangled_vit(tau=0.03):
    model = torchvision.models.vit_b_16(pretrained=True)
    layer_path = 'encoder.layers.encoder_layer_11.mlp.3'
    channel_to_disentangle = None
    model_dir = '/scratch/inf0/user/dbagci/vitb16/models_disentangled'
    if tau==0.03:
        channel_to_disentangle = VIT_MULTI_POLY_CHANNELS_0_03
        model_dir = '/scratch/inf0/user/dbagci/vitb16/models_disentangled_0.03'
    elif tau==0.02:
        channel_to_disentangle = VIT_MULTI_POLY_CHANNELS_0_02
        model_dir = '/scratch/inf0/user/dbagci/vitb16/models_disentangled'
    elif tau==0.01:
        model_dir = '/scratch/inf0/user/dbagci/vitb16/models_disentangled_0.01/models_disentangled_0.01'
        channel_to_disentangle = VIT_MULTI_POLY_CHANNELS_0_01
    else:
        assert False, "Tau not available"

    model2 = copy.deepcopy(model).to("cuda:0")
    random_input = torch.randn((1,3,224,224)).to("cuda:0")
    for entry in channel_to_disentangle:
        set_of_classes = channel_to_disentangle[entry]

        model = load_multi_disentangled_vit(model, layer_path, int(entry), set_of_classes, model_dir=model_dir).to("cuda:0")
        print('Are equal:')
        print(torch.isclose(model(random_input), model2(random_input), atol=1e-03).all())
        #print(model.layer4[2].conv3[0])
        #print(model.layer4[2].conv3[0].weight.shape)
        layer_path = 'encoder.layers.encoder_layer_11.mlp.3' 
    
    return model, 'encoder.layers.encoder_layer_11.mlp.3'


def load_multi_disentangled_dinov2(model, layer_path, channel_of_interest, classes, loss, device="cuda:0", f=None, model_dir='/data/vimb06/resnet_store_dir/models_disentangled'):
    model = model.to(device)
    
    model_disentangled_final = copy.deepcopy(model)
    model_disentangled_final = model_disentangled_final.to(device)
    model_disentangled_final.eval()
    
    model_type = 'dinov2_base'
    #use_bias = False if "supervised" in loss else True
    model_disentangled = MultiClassTrainingModelViT(channel_of_interest, 3072, num_classes=len(classes), bias=True).to(device)
    model_disentangled = model_disentangled.to(device)

    f = f
    if f == None:
        f = '_best'

    model_disentangled.load_state_dict(torch.load(os.path.join(model_dir, model_type + '_channel' + str(channel_of_interest) + '_f' + str(f) + '_unnormalized.pth')), strict=False)
    model_layer = get_layer_by_path(model, layer_path)
    iden = None

    if isinstance(model_layer, nn.Sequential):
        iden = model_layer[1]
        model_layer = model_layer[0]
    in_channels, out_channels = model_layer.in_features, model_layer.out_features

    model_disentangled_final_layer = get_layer_by_path(model_disentangled_final, layer_path)
    model_disentangled_final_layer = nn.Sequential(
            nn.Linear(3072, out_channels+len(classes), bias=True),
            nn.Linear(out_channels+len(classes), 768, bias=False)
    ).to(device)

    new_weight_Lmin1_L = model_layer.weight.clone() #model.layer4[2].conv3.weight.clone()

    new_weight_Lmin1_L[channel_of_interest, :] = model_disentangled.disentangled.weight[0,:].clone()
    for i in range(1, len(classes)+1):
        new_weight_Lmin1_L = torch.cat([new_weight_Lmin1_L,model_disentangled.disentangled.weight[i:i+1,:].clone()], dim=0) 

    model_disentangled_final_layer[0].weight = nn.Parameter(new_weight_Lmin1_L)
    
    if model_layer.bias is not None:
        new_bias = model_layer.bias.clone()
        # Add bias for the new disentangled neurons (len(classes) new neurons total)
        # All new neurons get the bias from the original channel_of_interest
        for i in range(len(classes)):
            new_bias = torch.cat([new_bias, model_layer.bias[channel_of_interest:channel_of_interest+1].clone()], dim=0)
        model_disentangled_final_layer[0].bias = nn.Parameter(new_bias)
    
    if not iden is None:
        prev_identity = iden.weight.clone()  # Copy previous identity matrix
        identity = torch.zeros((768, out_channels+len(classes))).to(device)
        identity[:prev_identity.shape[0], :prev_identity.shape[1]] = prev_identity
    else:
        identity = torch.zeros((768, out_channels+len(classes))).to(device)
        for i in range(out_channels):
            identity[i,i] = 1.

    for i in range(len(classes)):
        identity[channel_of_interest,out_channels+i] = 1. 
    
    identity[channel_of_interest,out_channels+len(classes)-1] = -1. * (len(classes)-1)
    #model_disentangled_final.layer4[2].conv3[1].weight = nn.Parameter(identity)
    model_disentangled_final_layer[1].weight = nn.Parameter(identity)
    set_layer_by_path(model_disentangled_final, layer_path, model_disentangled_final_layer)
    return model_disentangled_final


def load_fully_multi_disentangled_dinov2(model, loss="supervised", tau=0.03):
    layer_path = "blocks.11.mlp.fc2" if not loss == "supervised" else "backbone.blocks.11.mlp.fc2"
    channel_to_disentangle = None
    model_dir = "/scratch/inf0/user/dbagci/dinov2_base"
    if loss == "supervised":
        if tau==0.015:
            channel_to_disentangle = {"688": [386, 520, 653, 658, 659, 662, 791, 790, 537, 927, 38, 43, 300, 429, 302, 559, 690, 951, 311, 825, 576, 833, 71, 967, 591, 470, 857, 988, 993, 739, 868, 488, 489, 748, 365, 236, 111, 880, 498, 756, 117, 502], "249": [362, 476, 277], "431": [131, 340], "550": [416, 397], "620": [128, 134], "601": [400, 907, 317], "2": [293, 21, 22, 87, 88, 375], "96": [120, 931], "375": [861, 990], "131": [41, 83, 613], "80": [139, 7, 143], "11": [89, 51], "518": [628, 311], "670": [376, 174, 510], "750": [193, 527], "662": [547, 396, 15, 277, 863], "648": [113, 306, 314, 573], "629": [358, 719], "193": [800, 441, 867], "270": [993, 307], "89": [139, 31], "426": [154, 366], "642": [48, 986, 388], "377": [755, 102], "336": [225, 516], "93": [625, 429], "304": [141, 270], "300": [280, 755], "194": [984, 974], "318": [211, 694, 739], "419": [707, 316, 293], "152": [776, 800, 135], "523": [169, 887], "342": [364, 607], "611": [974, 318], "188": [76, 92], "728": [72, 221], "665": [524, 70], "107": [290, 331], "520": [616, 410, 510], "345": [430, 311], "136": [992, 47], "13": [100, 661], "514": [218, 211, 564], "503": [338, 171, 739], "502": [579, 14, 7], "687": [297, 354], "691": [234, 100], "562": [129, 403], "287": [802, 292], "740": [277, 575], "548": [691, 292], "566": [297, 430], "231": [993, 422], "417": [130, 847], "305": [505, 387], "115": [792, 510]}
            model_dir = "/scratch/inf0/user/dbagci/dinov2_base/models_disentangled_0.015/"#"/BS/disentanglement/work/unsupervised/dinov2/clustering_0/adaptive_kmeans/models_disentangled_0.02_recon_loss/"
        elif tau==0.02:
            channel_to_disentangle = {"688": [833, 386, 868, 38, 71, 748, 791, 951, 659, 790, 311], "2": [21, 22, 375], "188": [76, 92], "642": [48, 388], "417": [130, 847]}
            model_dir = "/scratch/inf0/user/dbagci/dinov2_base/models_disentangled_0.02/"
            #            /scratch/inf0/user/dbagci/dinov2_base/models_disentangled_0.02/dinov2_base_channel688_f_best_unnormalized.pth
            #            /scratch/inf0/user/dbagci/dinov2_base/models_disentangled_0.02/dinov2_channel688_f_best_unnormalized.pth
        elif tau==0.01:
            channel_to_disentangle = {"688": [520, 8, 534, 537, 27, 28, 549, 38, 551, 43, 556, 559, 561, 52, 573, 575, 576, 71, 591, 601, 111, 115, 117, 631, 126, 651, 653, 654, 658, 659, 660, 662, 690, 711, 739, 748, 236, 756, 244, 759, 760, 761, 771, 777, 778, 269, 275, 787, 790, 791, 798, 803, 805, 300, 302, 814, 310, 311, 825, 831, 832, 833, 846, 857, 858, 862, 865, 868, 877, 365, 879, 880, 885, 894, 896, 385, 386, 912, 914, 926, 927, 931, 421, 935, 429, 432, 434, 947, 951, 450, 967, 969, 971, 463, 470, 475, 988, 478, 993, 488, 489, 498, 502], "268": [903, 652, 529, 276, 922, 796, 802, 419, 549, 808, 819, 692, 838, 968, 457, 714, 591, 465, 340, 728, 478, 741, 110, 111, 114, 630], "2": [140, 396, 21, 22, 23, 546, 293, 550, 42, 47, 48, 303, 50, 306, 61, 466, 340, 87, 88, 353, 355, 746, 372, 375], "317": [684, 686, 632, 125, 30], "601": [130, 451, 328, 907, 140, 496, 400, 822, 317], "78": [260, 710, 797, 428, 332, 919, 825, 61], "228": [192, 129, 2, 869, 936, 233, 266, 173, 142, 208, 182, 186, 189], "193": [800, 867, 100, 712, 111, 402, 443, 441, 27], "415": [674, 451, 916, 421], "117": [928, 929, 256, 515, 102, 967, 363, 15, 208, 51, 596, 598, 759, 248, 250, 796, 477], "249": [995, 362, 277, 87, 889, 476], "466": [800, 97, 779, 560, 337, 370, 598, 854, 892, 477, 62], "616": [517, 710, 170, 299, 107, 563, 218], "318": [739, 803, 838, 168, 522, 170, 621, 211, 694, 279, 254], "375": [742, 551, 424, 585, 346, 335, 981, 250, 956, 861, 990], "497": [640, 994, 388, 710, 364, 76, 467, 917, 727, 731], "52": [417, 2, 679, 618, 112, 91], "593": [736, 227, 776, 616, 169, 652, 275, 822, 345, 541], "222": [256, 385, 228, 297, 560, 177, 797, 446], "336": [225, 516, 747, 239, 723, 603, 316, 255], "103": [387, 516, 19, 411, 443, 957], "189": [122, 709, 670], "91": [344, 794], "304": [768, 833, 139, 141, 270, 339, 279], "6": [0, 130, 739, 869, 134, 654, 496, 661], "171": [227, 900, 39, 712, 392, 263, 171, 237, 177, 245, 444, 158, 191], "356": [612, 971, 148, 729, 251], "342": [610, 364, 113, 145, 607], "750": [193, 167, 810, 527, 184], "513": [612, 42, 108, 77, 603], "164": [679, 333, 878, 766, 398, 343, 348, 702], "682": [388, 102, 10, 11, 13, 946, 19, 29, 154, 222, 926], "109": [741, 41, 781, 15, 145, 92], "431": [131, 295, 647, 171, 686, 177, 340, 757], "131": [225, 613, 39, 41, 366, 81, 83], "156": [288, 668, 226, 388, 72, 368, 176, 592, 275, 212, 21, 218, 92], "687": [128, 354, 547, 297, 270, 367, 307, 375, 249], "555": [129, 463, 570, 25, 986, 667, 284, 125], "602": [48, 377, 786], "93": [262, 136, 429, 145, 817, 625, 470], "312": [81, 894, 54], "670": [510, 174, 471, 376, 574], "316": [467, 316, 71], "642": [385, 1, 388, 296, 48, 149, 89, 986], "172": [318, 98, 103, 71, 74, 17, 373, 343, 382], "711": [160, 834, 995, 197, 869, 199, 853], "550": [96, 417, 416, 106, 397, 597], "365": [288, 449, 74, 623, 503, 28], "691": [739, 100, 234, 110, 957, 114, 116, 88, 317], "489": [65, 739, 300, 114, 115, 990], "303": [454, 778, 939, 77, 467, 468, 565, 215, 156, 607], "525": [355, 356, 938, 563, 52, 604], "648": [96, 291, 218, 113, 306, 690, 214, 314, 187, 573], "101": [81, 506, 331, 117], "367": [352, 131, 136, 143, 19, 117, 123], "50": [651, 144, 145, 498, 534], "167": [450, 521, 490, 111, 501], "745": [423, 455, 841, 658, 631], "412": [640, 865, 396, 878, 595, 308, 986, 31], "514": [8, 206, 211, 564, 213, 215, 217, 218], "79": [256, 995, 139, 715, 843, 142, 146, 87, 221, 94], "634": [236, 142, 191], "377": [387, 102, 755, 599, 61], "253": [385, 91, 15], "13": [576, 100, 368, 661, 59, 126], "121": [114, 451, 500, 318], "349": [40, 820, 829, 991], "380": [962, 883, 543], "289": [100, 196, 167, 172, 241, 822], "298": [70, 106, 43, 52, 247], "605": [707, 872, 938, 365, 55, 920], "564": [5, 594, 694, 184, 185, 475, 252], "662": [98, 547, 39, 937, 938, 396, 15, 466, 277, 863, 123, 351], "654": [280, 337, 330, 474], "620": [128, 289, 134, 649, 498], "575": [240, 776, 434, 498], "85": [33, 149, 814], "720": [35, 69, 654], "592": [974, 787, 148, 147, 959], "457": [416, 291, 297, 622, 276, 670], "385": [208, 209, 332, 238], "191": [274, 871], "520": [896, 616, 520, 139, 410, 510, 287], "38": [882, 387, 477], "136": [992, 694, 38, 47], "372": [225, 388, 230, 391, 563, 63], "192": [944, 976, 210, 339, 243], "561": [128, 833, 98, 235], "518": [576, 258, 291, 69, 628, 22, 311], "767": [32, 88, 287], "263": [482, 99, 75, 141, 276, 732], "325": [48, 296, 368], "502": [579, 7, 45, 14, 724], "199": [616, 398], "629": [358, 719, 335, 915, 91, 863], "733": [484, 232, 786, 341, 669], "122": [96, 168, 173, 206, 690], "759": [707, 940, 140, 83, 435, 410, 284, 703], "71": [673, 205, 703], "382": [96, 39, 560, 574, 863], "421": [75, 683, 779, 369, 786, 95], "584": [736, 866, 484, 550, 843, 886], "243": [466, 103], "700": [85, 46, 95], "413": [64, 385, 38], "410": [59, 43, 365], "460": [327, 141, 334, 57, 378, 221], "285": [320, 31, 948, 956, 319], "215": [131, 547, 232, 72, 210, 382, 383], "181": [753, 186, 982, 847], "351": [392, 313], "270": [993, 74, 307, 251], "240": [513, 690, 935], "764": [145, 947], "511": [138, 45, 134], "1": [120, 465, 994], "598": [251, 28], "467": [466, 116, 244], "477": [832, 365, 431, 242, 184], "264": [130, 554, 561, 147, 189, 190], "671": [614, 371, 597, 377, 378], "194": [140, 12, 974, 944, 245, 984, 601, 607], "41": [176, 405], "419": [707, 293, 169, 245, 316], "465": [938, 594, 39, 914], "677": [161, 162, 450, 685, 116, 342, 607], "96": [931, 300, 561, 401, 120], "213": [80, 83, 165], "481": [198, 718, 143, 240, 56, 379], "43": [340, 293, 7], "73": [24, 9, 87], "656": [225, 525], "724": [881, 765, 31], "704": [43, 276, 863, 351], "516": [258, 228, 247], "216": [236, 13, 209, 535, 376, 861], "490": [1, 335, 791], "515": [290, 548, 102, 343, 24], "397": [770, 259, 788, 71], "195": [89, 18], "676": [802, 230, 351, 798, 286], "345": [80, 280, 430, 311], "717": [993, 132, 333, 566, 316, 29], "562": [225, 129, 131, 403], "547": [259, 388, 333, 565, 598], "653": [521, 645], "357": [865, 635, 196, 269], "739": [147, 932, 317], "80": [0, 7, 393, 139, 143], "276": [305, 537], "89": [640, 72, 139, 751, 242, 116, 349, 31], "596": [256, 57, 481, 653], "34": [88, 168], "646": [136, 729, 363], "586": [80, 24, 10], "11": [89, 290, 51, 550], "315": [137, 28, 933, 881], "197": [48, 651, 268], "505": [688, 425], "523": [169, 490, 881, 887, 26, 669], "36": [41, 275, 236, 405], "715": [97, 332, 5, 935], "203": [545, 99, 614, 400, 540], "667": [194, 839, 392, 138, 407, 190], "88": [320, 8, 144, 86, 253], "548": [672, 292, 998, 691, 543], "68": [611, 187, 142], "686": [448, 83, 820, 279, 319], "368": [292, 30], "728": [72, 146, 221], "292": [144, 481, 18, 565], "527": [875, 79, 372, 820, 24], "218": [878, 180, 30], "725": [128, 233, 547], "65": [340, 199], "422": [318, 604, 70], "695": [422, 71, 7, 206, 821], "130": [192, 258, 132], "226": [140, 20, 278, 95], "115": [0, 338, 792, 281, 510], "506": [577, 297], "763": [881, 45, 230, 95], "252": [65, 71], "469": [49, 451, 785], "408": [271, 87, 151], "224": [81, 129, 84], "714": [586, 109, 934, 695], "20": [704, 393, 520, 79], "104": [56, 474, 572], "528": [450, 100, 364, 142, 307], "259": [49, 881, 7], "24": [237, 375], "280": [486, 182, 281, 27, 350], "600": [104, 875, 317, 343], "694": [354, 219, 550, 739], "526": [493, 743], "154": [91, 333, 294, 319], "606": [136, 183, 318, 231], "448": [416, 289, 93, 150], "123": [513, 300, 145, 625, 286], "411": [330, 523, 658], "622": [9, 821, 510], "379": [992, 974, 144, 369, 123, 127, 383], "188": [33, 76, 92, 167], "299": [312, 396], "693": [408, 915, 374], "269": [292, 149, 639], "463": [981, 545, 645, 918], "391": [438, 679], "539": [261, 389, 203, 174, 156], "245": [81, 330, 27], "631": [776, 53, 383], "281": [3, 955, 949], "373": [12, 870, 199], "450": [57, 203], "234": [513, 14, 81, 82, 561], "163": [508, 574, 423], "260": [377, 247], "279": [290, 123, 157], "301": [306, 181], "110": [256, 164, 356], "81": [171, 174], "658": [937, 524, 283, 91, 286], "178": [152, 160, 79], "182": [365, 389], "579": [858, 195, 707], "106": [25, 15], "210": [320, 563, 76, 182], "147": [732, 78, 959], "512": [966, 239], "77": [283, 340, 484, 847], "187": [476, 36], "674": [789, 486], "747": [292, 261, 136, 236, 116, 351], "407": [176, 19, 995], "587": [988, 444, 39, 495], "396": [129, 125, 983], "748": [122, 139, 351], "92": [576, 292, 80, 726, 958], "753": [343, 183], "344": [364, 294], "478": [281, 278], "433": [336, 386, 261, 343], "438": [98, 475, 46], "663": [24, 580], "731": [128, 336], "597": [178, 388], "150": [264, 617, 715, 13, 214], "17": [306, 900, 486], "278": [330, 300, 332, 272, 85], "445": [560, 918], "709": [833, 420, 366], "231": [993, 422, 333, 816, 244], "335": [293, 174], "99": [239, 566, 135], "239": [91, 983], "744": [602, 555], "420": [43, 76], "287": [299, 802, 195, 292], "326": [548, 425, 713, 685, 48, 400], "127": [37, 334], "166": [992, 289, 594, 963], "331": [736, 531, 524, 685], "451": [57, 83, 533, 382], "738": [161, 306, 164, 284], "471": [49, 121, 486], "426": [154, 875, 116, 366], "271": [142, 37, 30, 467], "692": [955, 667, 21], "128": [129, 122, 92], "638": [280, 242, 100, 278], "53": [952, 399], "84": [425, 106, 268], "328": [832, 137, 195], "542": [986, 199], "698": [33, 658, 401], "295": [362, 243, 900, 933], "756": [944, 219, 991], "248": [640, 719], "417": [945, 130, 286, 847], "655": [994, 995, 959], "543": [385, 291, 668, 974], "402": [800, 995, 125], "386": [385, 133, 991], "439": [272, 305, 847], "3": [593, 387], "302": [13, 933], "66": [833, 244], "640": [944, 84, 704], "49": [120, 757, 607], "427": [702, 102, 967], "625": [202, 157, 378, 191], "300": [280, 755], "120": [297, 79], "173": [164, 92], "765": [609, 795, 262], "665": [70, 524, 238, 791, 889], "552": [209, 27], "22": [780, 77], "453": [416, 181, 303], "67": [160, 291], "446": [331, 227], "619": [17, 131, 47], "660": [88, 528, 802], "168": [10, 974], "361": [249, 723], "319": [824, 92], "273": [215, 547, 167], "751": [138, 604], "480": [347, 171, 396], "159": [736, 403, 143], "668": [640, 759], "712": [120, 403], "55": [0, 72, 699, 873], "563": [338, 291, 20, 373], "35": [162, 139, 687], "152": [800, 135, 776, 10, 607], "633": [236, 165, 102], "705": [604, 349, 239], "678": [29, 957], "257": [360, 439], "359": [466, 839], "583": [289, 986, 196], "757": [171, 229], "536": [112, 559], "72": [283, 100], "755": [155, 108], "352": [919, 287], "155": [320, 289, 666, 467], "459": [763, 276, 189], "401": [537, 138, 423], "664": [50, 214], "503": [338, 171, 989, 739], "754": [243, 335], "254": [665, 27, 36], "297": [376, 294], "644": [945, 779, 886], "614": [393, 242, 347], "142": [874, 327], "314": [362, 131], "566": [297, 715, 430], "392": [277, 199], "235": [261, 670], "458": [122, 284, 573, 566], "636": [296, 302], "464": [724, 95], "353": [364, 375, 367, 87], "294": [927, 157, 271], "146": [376, 513], "681": [537, 38], "64": [112, 290, 134], "521": [739, 438], "488": [289, 87], "493": [915, 316], "627": [945, 866, 383], "442": [376, 139], "485": [719, 342, 87], "108": [160, 933], "355": [294, 212, 382], "611": [520, 974, 318, 431], "45": [816, 540, 886], "347": [688, 137], "169": [336, 607], "290": [218, 43, 378, 763], "517": [128, 336, 396, 127], "140": [74, 135], "551": [385, 91], "534": [16, 191], "679": [713, 929], "399": [665, 69, 670], "114": [656, 260, 261], "703": [992, 286], "434": [201, 820], "33": [369, 139], "499": [974, 127], "559": [401, 12], "107": [561, 290, 331], "134": [217, 476], "498": [934, 687], "212": [104, 268], "476": [466, 286], "153": [112, 694], "185": [161, 135], "726": [896, 992, 294], "47": [500, 278, 300], "409": [387, 788, 261], "741": [576, 933, 510], "462": [736, 357], "8": [165, 95], "393": [704, 105], "376": [9, 396], "12": [275, 142], "9": [393, 971], "305": [387, 486, 138, 505, 667], "124": [105, 540, 141], "237": [57, 243], "740": [891, 277, 575], "697": [666, 141, 726], "560": [346, 254], "590": [752, 537, 847], "217": [284, 535], "132": [226, 746, 890], "680": [889, 820, 174, 255], "675": [352, 254], "531": [17, 319], "533": [43, 758], "209": [320, 183], "339": [849, 108, 604], "719": [251, 383], "487": [192, 7], "255": [16, 66], "250": [968, 260], "378": [739, 541, 607], "113": [932, 614], "90": [779, 886], "387": [410, 133], "369": [606, 575], "666": [182, 863], "637": [251, 143], "138": [763, 847], "4": [289, 197], "26": [377, 963], "389": [84, 510], "208": [576, 22], "284": [833, 11], "16": [400, 667, 791], "418": [224, 961, 400], "443": [11, 300], "366": [547, 299, 981], "730": [875, 755], "452": [528, 307], "251": [776, 275], "461": [456, 881], "296": [401, 101], "447": [112, 528], "734": [820, 886]}
            model_dir = "/scratch/inf0/user/dbagci/dinov2_base/models_disentangled_0.01/"
        else:
            assert False, "Tau not available"
    elif loss == "IG":
        if tau == 0.02:
            channel_to_disentangle = {"3": [297, 285], "38": [0, 513, 516, 5, 7, 519, 521, 522, 523, 12, 525, 14, 15, 528, 529, 18, 527, 535, 536, 537, 25, 27, 28, 30, 543, 32, 544, 40, 41, 553, 43, 557, 45, 47, 559, 561, 564, 55, 56, 568, 570, 59, 62, 64, 577, 67, 70, 71, 587, 589, 590, 80, 592, 596, 89, 602, 601, 94, 606, 96, 611, 614, 615, 106, 619, 618, 621, 622, 625, 115, 116, 117, 118, 119, 632, 634, 123, 635, 641, 646, 647, 649, 650, 139, 652, 654, 656, 657, 149, 667, 156, 669, 158, 671, 672, 670, 674, 164, 677, 679, 680, 681, 170, 683, 685, 175, 176, 687, 691, 180, 181, 694, 695, 185, 186, 697, 702, 703, 704, 707, 196, 199, 200, 716, 717, 205, 719, 209, 721, 723, 725, 730, 731, 732, 734, 735, 736, 228, 743, 232, 745, 746, 238, 239, 753, 246, 247, 762, 763, 252, 766, 768, 257, 771, 261, 774, 263, 775, 776, 779, 268, 781, 270, 782, 273, 788, 789, 277, 791, 278, 793, 281, 794, 796, 792, 801, 807, 809, 299, 307, 309, 311, 314, 315, 318, 326, 332, 340, 342, 343, 349, 351, 354, 356, 360, 363, 365, 371, 374, 375, 378, 380, 381, 383, 384, 387, 391, 393, 394, 396, 403, 404, 405, 416, 422, 423, 424, 426, 434, 435, 438, 442, 453, 462, 463, 465, 467, 469, 471, 472, 478, 485, 486, 487, 489, 490, 491, 492, 493, 496, 497, 501, 506, 508, 511], "53": [440, 407], "60": [728, 152, 424], "71": [105, 109], "91": [523, 203, 6], "101": [0, 202], "107": [53, 150], "117": [513, 521, 13, 526, 18, 25, 31, 551, 553, 560, 561, 562, 565, 54, 568, 572, 64, 582, 588, 589, 592, 596, 87, 601, 92, 606, 614, 621, 625, 116, 633, 634, 636, 125, 653, 148, 667, 669, 674, 164, 677, 679, 175, 695, 186, 703, 192, 704, 707, 712, 205, 717, 725, 735, 736, 232, 746, 748, 762, 763, 766, 771, 263, 776, 272, 796, 291, 305, 338, 339, 354, 369, 381, 392, 394, 396, 405, 422, 447, 456, 467, 491, 501, 508, 511], "122": [673, 5, 574, 585, 238, 239, 401, 533, 569, 381, 254, 415], "126": [401, 633], "127": [609, 267], "156": [742, 680, 240, 569, 798, 639], "161": [640, 641, 225, 770, 36, 548, 546, 455, 464, 691, 531, 51, 566, 567, 121, 540], "164": [773, 397], "177": [681, 15, 143], "216": [2, 517, 12, 526, 527, 538, 540, 544, 34, 546, 39, 552, 44, 46, 562, 565, 567, 569, 575, 579, 69, 584, 585, 74, 599, 601, 91, 604, 608, 104, 616, 618, 620, 624, 114, 630, 121, 635, 637, 126, 640, 642, 645, 137, 650, 138, 649, 145, 659, 664, 158, 670, 672, 162, 677, 684, 178, 187, 701, 702, 194, 706, 710, 723, 725, 729, 222, 742, 231, 744, 743, 745, 752, 241, 244, 756, 758, 760, 248, 762, 251, 254, 257, 769, 771, 774, 265, 269, 781, 271, 782, 787, 276, 789, 277, 279, 790, 792, 286, 800, 289, 802, 804, 806, 304, 309, 319, 321, 326, 337, 349, 366, 367, 376, 389, 390, 398, 399, 400, 401, 405, 409, 410, 416, 417, 418, 419, 420, 424, 427, 428, 434, 438, 442, 444, 445, 453, 457, 462, 463, 465, 466, 472, 476, 489, 495, 498, 499, 500, 503, 505], "228": [69, 778, 16, 688, 376, 444], "246": [680, 804], "268": [0, 513, 771, 774, 263, 776, 393, 649, 779, 396, 523, 13, 521, 272, 18, 790, 796, 28, 669, 800, 674, 419, 804, 677, 679, 553, 42, 561, 568, 570, 702, 703, 704, 707, 582, 71, 713, 717, 205, 592, 340, 725, 596, 87, 601, 218, 731, 220, 94, 735, 736, 611, 762, 614, 232, 745, 746, 491, 621, 625, 116, 380, 634, 635, 508, 766], "293": [0, 9, 12, 18, 535, 544, 42, 46, 565, 570, 575, 74, 75, 591, 81, 601, 94, 95, 608, 611, 615, 104, 116, 122, 636, 127, 643, 644, 645, 657, 666, 667, 671, 677, 684, 178, 692, 699, 705, 707, 712, 200, 713, 722, 729, 733, 222, 228, 743, 234, 751, 244, 765, 262, 775, 776, 777, 283, 808, 317, 326, 337, 340, 344, 349, 353, 354, 367, 405, 420, 436, 443, 454, 456, 467, 488, 499], "354": [270, 790], "390": [231, 797, 728, 29, 415], "408": [578, 692, 37, 95], "415": [768, 641, 769, 782, 272, 792, 796, 418, 419, 548, 674, 806, 802, 546, 686, 561, 51, 691, 566, 568, 703, 575, 582, 456, 589, 464, 465, 467, 725, 608, 736, 618, 619, 364, 621, 239, 760, 376, 762, 254], "423": [585, 153], "436": [516, 607], "460": [449, 201, 727], "485": [216, 275], "497": [0, 261, 70, 774, 424, 521, 496, 596, 94, 601, 570, 732, 702, 703], "516": [120, 477], "528": [753, 301], "555": [629, 327], "568": [768, 7, 273, 18, 532, 798, 542, 671, 808, 424, 426, 558, 559, 690, 694, 698, 203, 590, 725, 343, 750, 248], "572": [741, 136, 652, 593, 598, 727], "623": [714, 260], "642": [280, 705, 393, 68], "645": [517, 518, 12, 527, 529, 537, 546, 35, 548, 552, 51, 566, 572, 582, 586, 589, 591, 602, 603, 92, 609, 610, 619, 621, 111, 113, 121, 123, 637, 126, 641, 655, 668, 687, 688, 179, 691, 183, 185, 702, 717, 719, 721, 726, 734, 739, 740, 232, 235, 747, 754, 757, 245, 248, 768, 778, 781, 792, 797, 802, 807, 302, 311, 312, 319, 331, 341, 345, 374, 379, 383, 406, 412, 416, 419, 421, 439, 450, 452, 464, 465, 471, 491, 495], "661": [110, 591], "687": [578, 229, 38, 808, 429, 206, 632], "688": [0, 512, 5, 518, 521, 522, 525, 14, 15, 16, 18, 533, 24, 27, 28, 542, 32, 42, 557, 45, 50, 567, 568, 570, 58, 577, 582, 587, 75, 589, 79, 593, 83, 596, 603, 98, 611, 108, 621, 110, 125, 643, 656, 662, 681, 683, 686, 698, 700, 703, 707, 710, 713, 725, 218, 220, 733, 741, 750, 239, 238, 240, 243, 762, 763, 765, 255, 768, 259, 776, 778, 779, 780, 270, 272, 273, 793, 281, 798, 810, 300, 314, 339, 344, 354, 356, 376, 380, 403, 413, 422, 443, 456, 460, 474, 478, 491], "706": [521, 123, 94], "708": [770, 392, 781, 792, 797, 672, 548, 427, 51, 574, 588, 205, 206, 462, 464, 471, 730, 221, 737, 225, 743, 361, 618, 619, 240, 497, 630, 762], "715": [225, 69], "721": [258, 261, 638, 782, 655, 401, 530, 535, 410, 540, 801, 807, 428, 559, 565, 693, 567, 569, 332, 341, 602, 99, 103, 624, 637, 254], "733": [610, 623], "743": [601, 67, 627, 521], "756": [497, 172, 662, 361]}
            model_dir = "/BS/disentanglement/work/unsupervised/dinov2_base/clustering_0_0.5/adaptive_kmeans/models_disentangled_0.02_IG"
        elif tau == 0.015:
            channel_to_disentangle = {"2": [288, 113, 234, 468], "3": [297, 285], "7": [352, 269], "9": [640, 120], "11": [298, 35, 157], "16": [246, 175], "19": [479, 127], "20": [241, 481], "21": [290, 179], "23": [206, 627, 728, 766, 799], "27": [279, 79], "30": [56, 665, 258, 574], "38": [0, 513, 516, 5, 7, 519, 521, 522, 523, 524, 525, 14, 15, 528, 529, 18, 12, 527, 535, 536, 537, 25, 27, 28, 30, 543, 32, 544, 548, 40, 41, 553, 43, 557, 45, 47, 559, 48, 561, 564, 55, 56, 568, 570, 59, 62, 63, 64, 577, 67, 70, 71, 74, 587, 588, 589, 590, 79, 80, 592, 596, 89, 602, 91, 601, 606, 94, 96, 611, 614, 615, 106, 619, 618, 109, 622, 621, 625, 115, 116, 117, 118, 119, 632, 634, 123, 635, 641, 646, 647, 649, 650, 139, 652, 654, 656, 657, 149, 664, 667, 156, 669, 158, 671, 672, 670, 674, 164, 677, 679, 680, 681, 170, 683, 685, 175, 176, 687, 691, 180, 181, 694, 695, 185, 186, 697, 702, 703, 704, 705, 707, 196, 199, 200, 201, 716, 717, 205, 719, 209, 721, 723, 725, 730, 731, 732, 734, 735, 736, 228, 743, 232, 745, 746, 238, 239, 753, 246, 247, 760, 761, 762, 763, 252, 766, 768, 257, 771, 261, 774, 263, 775, 776, 779, 268, 781, 270, 782, 273, 788, 789, 277, 791, 278, 793, 281, 794, 796, 284, 792, 801, 807, 809, 810, 299, 307, 309, 311, 314, 315, 318, 326, 328, 332, 338, 340, 342, 343, 349, 351, 354, 356, 360, 363, 365, 371, 374, 375, 378, 379, 380, 381, 383, 384, 387, 391, 393, 394, 396, 403, 404, 405, 416, 422, 423, 424, 426, 434, 435, 438, 441, 442, 444, 453, 462, 463, 465, 467, 469, 471, 472, 478, 485, 486, 487, 489, 490, 491, 492, 493, 496, 497, 501, 506, 508, 511], "39": [344, 750], "42": [409, 346, 230], "45": [67, 54], "52": [768, 746, 273, 82, 446], "53": [576, 290, 770, 623, 466, 407, 440, 313], "56": [84, 101], "58": [571, 102], "60": [424, 469, 728, 668, 152, 671], "61": [35, 151], "71": [105, 109], "76": [148, 565, 526], "77": [198, 135], "78": [105, 458, 387, 164], "80": [583, 168, 431, 411, 190], "84": [48, 681, 478], "87": [53, 207], "89": [417, 554, 678], "91": [131, 516, 6, 393, 394, 523, 396, 664, 537, 793, 543, 548, 554, 170, 686, 703, 705, 203, 596, 736], "95": [136, 321, 86], "99": [54, 367], "101": [0, 202, 203, 756, 404, 150, 798], "107": [53, 150], "116": [101, 191], "117": [513, 521, 13, 526, 18, 25, 29, 31, 551, 553, 560, 48, 562, 561, 565, 54, 55, 568, 570, 572, 64, 576, 582, 74, 588, 589, 592, 596, 87, 601, 92, 94, 606, 614, 621, 625, 116, 630, 633, 634, 636, 125, 653, 654, 148, 149, 667, 669, 674, 164, 677, 679, 175, 695, 696, 186, 703, 192, 704, 707, 712, 716, 717, 205, 725, 218, 734, 735, 736, 739, 232, 746, 748, 762, 763, 766, 771, 263, 776, 272, 796, 291, 299, 305, 315, 338, 339, 354, 369, 375, 377, 381, 392, 393, 394, 396, 402, 405, 411, 422, 423, 434, 447, 456, 467, 491, 497, 501, 508, 511], "122": [770, 5, 782, 401, 533, 286, 415, 32, 673, 177, 566, 567, 569, 700, 574, 585, 461, 717, 100, 238, 239, 624, 381, 254], "124": [51, 430], "126": [801, 103, 688, 401, 242, 633], "127": [609, 267], "129": [608, 723, 94, 511], "131": [296, 260, 621], "132": [513, 231], "133": [6, 759], "134": [704, 419, 456, 297, 393, 555, 747, 272, 561, 434, 725, 568], "136": [155, 324, 292], "139": [234, 260, 663], "141": [691, 220, 788], "154": [739, 191], "156": [608, 771, 255, 742, 680, 424, 234, 639, 240, 595, 20, 25, 728, 569, 798, 575], "161": [640, 641, 770, 390, 782, 401, 531, 410, 540, 801, 546, 36, 805, 548, 691, 51, 437, 566, 567, 455, 464, 471, 730, 225, 360, 618, 497, 628, 376, 121, 637], "164": [753, 250, 773, 397], "167": [648, 763], "171": [457, 213], "172": [486, 73, 268, 14, 432, 597, 217, 698], "177": [98, 681, 143, 15, 346, 251], "178": [361, 402, 475], "189": [584, 424], "191": [620, 490, 132, 333], "192": [512, 143], "194": [648, 193, 235], "196": [72, 331], "198": [377, 346, 11, 191], "201": [217, 137], "206": [786, 291], "212": [193, 93], "215": [536, 53], "216": [0, 2, 517, 9, 12, 526, 527, 17, 532, 538, 540, 544, 34, 546, 39, 552, 41, 44, 46, 562, 563, 564, 565, 567, 568, 569, 575, 579, 69, 584, 585, 74, 81, 599, 601, 91, 604, 93, 605, 608, 98, 612, 613, 104, 616, 618, 620, 110, 624, 114, 630, 121, 635, 637, 126, 640, 642, 645, 137, 650, 138, 649, 145, 659, 663, 664, 665, 155, 158, 670, 672, 162, 677, 684, 178, 694, 186, 187, 701, 702, 704, 194, 706, 710, 200, 713, 723, 725, 729, 222, 742, 231, 744, 743, 745, 752, 241, 242, 244, 756, 758, 760, 248, 762, 251, 765, 254, 257, 769, 771, 774, 265, 781, 782, 269, 271, 787, 276, 789, 277, 279, 792, 790, 286, 799, 800, 289, 802, 804, 806, 304, 305, 309, 312, 318, 319, 321, 326, 328, 337, 349, 358, 364, 366, 367, 376, 378, 389, 390, 398, 399, 400, 401, 405, 409, 410, 412, 416, 417, 418, 419, 420, 424, 427, 428, 434, 438, 442, 444, 445, 450, 453, 457, 462, 463, 465, 466, 469, 472, 476, 483, 484, 489, 490, 494, 495, 496, 498, 499, 500, 503, 505], "217": [168, 508], "220": [316, 85], "224": [307, 671], "226": [8, 715, 292, 446], "228": [577, 69, 295, 681, 810, 778, 686, 688, 464, 16, 561, 49, 50, 376, 444, 764], "231": [120, 100], "235": [376, 627, 111], "240": [448, 323], "246": [680, 264, 804, 17], "250": [648, 475], "254": [683, 461, 262, 711], "261": [56, 35, 701, 311], "262": [648, 97], "263": [756, 174], "265": [756, 85, 727], "267": [738, 804, 309, 213, 789, 313], "268": [0, 513, 519, 521, 523, 12, 13, 527, 18, 538, 28, 543, 549, 553, 42, 560, 561, 568, 570, 582, 71, 591, 592, 596, 87, 601, 92, 94, 608, 611, 614, 621, 625, 116, 120, 634, 635, 649, 650, 654, 656, 664, 669, 674, 677, 679, 685, 687, 695, 702, 703, 704, 707, 713, 717, 205, 725, 729, 218, 731, 220, 734, 735, 736, 232, 745, 746, 762, 763, 766, 771, 262, 263, 775, 774, 776, 779, 272, 790, 792, 793, 796, 800, 804, 318, 338, 340, 354, 366, 375, 380, 393, 396, 419, 422, 444, 491, 498, 508], "273": [240, 26], "275": [740, 806, 582, 712, 175, 633], "277": [384, 678], "282": [587, 268, 103], "283": [441, 399], "284": [660, 47], "285": [687, 2, 22, 31], "287": [745, 43], "289": [588, 367, 497, 629, 698, 797, 735], "290": [266, 77, 223], "293": [0, 517, 9, 521, 12, 18, 535, 544, 552, 42, 46, 565, 570, 575, 74, 75, 591, 81, 594, 596, 601, 94, 95, 608, 611, 614, 615, 104, 620, 116, 122, 636, 127, 131, 644, 645, 643, 657, 662, 666, 667, 671, 677, 170, 684, 687, 689, 178, 692, 695, 699, 700, 705, 707, 708, 712, 200, 713, 719, 722, 729, 733, 222, 228, 743, 234, 751, 244, 756, 765, 262, 775, 776, 777, 783, 283, 797, 808, 301, 317, 326, 337, 340, 342, 344, 349, 353, 354, 357, 367, 398, 405, 406, 420, 427, 433, 436, 443, 454, 456, 467, 468, 476, 488, 494, 497, 499], "298": [382, 575], "300": [267, 598], "303": [552, 649, 400, 660, 629], "304": [301, 157], "307": [293, 534], "312": [795, 515, 230, 519], "318": [487, 459, 48, 598, 63], "321": [545, 177, 70], "323": [201, 645], "324": [208, 120], "330": [49, 757], "331": [744, 202, 732], "336": [657, 802, 724], "339": [387, 772, 278], "341": [72, 279], "343": [288, 511], "354": [481, 706, 67, 584, 270, 790, 503, 795, 446], "362": [419, 31], "365": [411, 54], "370": [775, 488, 809, 585, 494, 367, 143, 751, 88, 765, 95], "372": [360, 617, 91, 806], "374": [680, 53], "375": [713, 523, 727], "378": [19, 466, 115], "382": [468, 62, 599], "383": [128, 131], "386": [310, 102], "390": [353, 225, 643, 4, 770, 518, 231, 72, 415, 621, 305, 29, 728, 797, 670, 31], "393": [448, 272, 449], "401": [756, 764, 373, 534], "408": [480, 578, 185, 37, 709, 329, 169, 143, 692, 84, 630, 537, 797, 95], "414": [59, 342, 231], "415": [768, 641, 769, 775, 393, 779, 781, 782, 399, 272, 401, 275, 148, 790, 792, 281, 540, 796, 798, 418, 419, 548, 802, 806, 551, 552, 674, 546, 686, 561, 51, 691, 438, 566, 568, 569, 443, 444, 575, 703, 582, 456, 584, 75, 589, 464, 465, 467, 725, 87, 89, 606, 608, 736, 612, 618, 619, 364, 491, 621, 239, 624, 760, 501, 376, 762, 508, 254, 511], "418": [274, 342], "419": [36, 214], "423": [585, 153, 443], "424": [148, 285], "425": [693, 764, 85, 175], "431": [394, 413], "436": [290, 516, 4, 135, 810, 689, 211, 441, 607], "439": [433, 211], "444": [107, 765, 95], "445": [752, 324], "448": [331, 660], "451": [218, 348], "453": [626, 555], "460": [449, 805, 165, 741, 201, 652, 556, 727, 698], "461": [506, 189], "465": [488, 284], "467": [366, 174, 766, 23], "468": [520, 211, 143], "485": [216, 275], "486": [189, 750], "495": [600, 467, 340], "497": [0, 261, 774, 521, 424, 553, 570, 702, 703, 196, 70, 596, 601, 92, 732, 94, 736, 485, 615, 747, 496], "499": [372, 678], "508": [80, 355], "509": [603, 259, 325], "511": [509, 206, 303], "513": [353, 749], "514": [9, 4], "516": [120, 477], "520": [529, 267, 740, 630], "521": [198, 22], "522": [664, 433], "527": [364, 575], "528": [753, 76, 301, 143], "536": [675, 180, 439], "548": [65, 229, 439, 120, 377, 60], "555": [550, 327, 138, 43, 204, 586, 692, 629], "564": [785, 50], "566": [348, 126, 23], "567": [363, 37, 294], "568": [768, 515, 261, 7, 522, 268, 652, 14, 783, 270, 657, 18, 273, 532, 148, 536, 793, 792, 156, 542, 671, 798, 33, 34, 546, 678, 424, 808, 426, 557, 558, 559, 686, 690, 691, 694, 311, 698, 59, 203, 590, 463, 593, 467, 468, 725, 343, 471, 731, 348, 354, 483, 741, 621, 750, 238, 623, 248, 121, 764, 255], "572": [805, 741, 637, 136, 106, 652, 593, 273, 598, 727, 24, 125], "575": [777, 422, 431], "576": [22, 167], "583": [761, 236], "584": [664, 652, 54], "586": [356, 52], "593": [706, 775, 525, 19, 255], "597": [358, 407], "602": [602, 749], "603": [322, 229, 534], "608": [352, 333], "615": [114, 59], "617": [21, 182], "620": [498, 299], "621": [321, 718], "622": [352, 595], "623": [714, 260, 356, 261], "629": [763, 533], "634": [613, 72, 143, 689, 663], "638": [132, 60], "639": [394, 117, 638, 278], "641": [16, 649, 372, 757], "642": [705, 68, 393, 23, 280], "645": [517, 518, 12, 13, 527, 529, 530, 23, 537, 546, 35, 548, 552, 51, 566, 572, 582, 584, 586, 589, 591, 79, 87, 602, 603, 92, 609, 610, 618, 619, 108, 621, 110, 111, 113, 630, 121, 123, 637, 126, 641, 132, 649, 137, 653, 655, 148, 152, 668, 672, 175, 688, 687, 179, 691, 183, 185, 699, 702, 709, 717, 205, 719, 721, 726, 214, 221, 734, 223, 739, 740, 227, 232, 235, 747, 754, 243, 757, 245, 248, 768, 778, 781, 272, 277, 792, 797, 802, 807, 302, 311, 312, 319, 329, 331, 341, 345, 358, 371, 374, 379, 383, 406, 412, 416, 419, 421, 428, 437, 439, 450, 452, 464, 465, 471, 491, 495, 502], "648": [777, 715], "651": [416, 246], "657": [464, 51, 756, 476, 572], "658": [298, 183], "661": [186, 110, 591], "662": [328, 613], "667": [440, 481], "681": [563, 101], "682": [680, 293, 192], "684": [186, 251, 422], "687": [352, 578, 229, 38, 808, 429, 206, 112, 692, 567, 632], "688": [0, 512, 5, 518, 521, 522, 525, 14, 15, 16, 18, 533, 24, 27, 28, 542, 32, 553, 42, 557, 45, 50, 567, 568, 570, 58, 577, 582, 72, 587, 75, 589, 79, 81, 593, 83, 596, 87, 603, 98, 611, 108, 621, 110, 628, 632, 636, 125, 131, 643, 656, 148, 662, 666, 674, 681, 683, 686, 698, 700, 703, 707, 710, 713, 725, 727, 729, 218, 220, 733, 741, 746, 750, 239, 238, 240, 243, 762, 763, 765, 255, 768, 256, 767, 259, 776, 778, 779, 780, 270, 784, 273, 786, 272, 793, 281, 798, 810, 300, 314, 326, 339, 340, 344, 354, 356, 376, 380, 403, 413, 422, 443, 446, 456, 460, 474, 478, 486, 488, 491, 509], "689": [88, 737, 515], "693": [242, 236], "695": [578, 91], "698": [193, 34, 10], "699": [421, 749, 127], "701": [122, 749], "703": [331, 427, 21], "704": [228, 269, 655], "706": [521, 531, 123, 253, 94], "708": [770, 392, 12, 781, 270, 144, 279, 792, 797, 672, 802, 548, 806, 424, 427, 178, 691, 51, 56, 569, 574, 455, 73, 588, 205, 206, 718, 464, 465, 462, 84, 471, 728, 473, 730, 347, 221, 737, 225, 743, 361, 618, 619, 624, 497, 240, 630, 248, 762], "713": [473, 332], "715": [225, 219, 69], "721": [640, 258, 261, 782, 655, 401, 530, 788, 535, 410, 540, 801, 807, 428, 559, 563, 437, 565, 693, 567, 569, 332, 594, 341, 602, 347, 734, 99, 103, 234, 624, 754, 244, 118, 254, 637, 638], "725": [612, 726], "730": [104, 203], "733": [610, 590, 623], "735": [688, 470], "743": [0, 67, 805, 518, 521, 649, 112, 401, 658, 627, 601], "745": [58, 597, 199], "748": [226, 547], "751": [96, 452], "754": [9, 381], "756": [361, 172, 14, 623, 497, 501, 662], "762": [618, 563, 93, 206]}
            model_dir = "/BS/disentanglement/work/unsupervised/dinov2_base/clustering_0_0.5/adaptive_kmeans/models_disentangled_0.015_IG"
    #elif loss == "cos_sim":
    #    if tau == 0.02:
    #        channel_to_disentangle = {"139": [1600, 1996, 1292], "275": [1600, 1996], "409": [1379, 1646, 1424, 1685, 1754], "471": [872, 1009, 1424], "504": [1545, 1825], "718": [1545, 1714], "901": [1652, 1556], "1018": [1874, 943]}
    #        model_dir = "/BS/disentanglement/work/unsupervised/dinov2/clustering_0/adaptive_kmeans/models_disentangled_0.02_cos_sim/"


    model2 = copy.deepcopy(model).to("cuda:0")
    random_input = torch.randn((1,3,224,224)).to("cuda:0")
    for entry in channel_to_disentangle:
        set_of_classes = channel_to_disentangle[entry]

        model = load_multi_disentangled_dinov2(model, layer_path, int(entry), set_of_classes, loss=loss, model_dir=model_dir).to("cuda:0")
        print('Are equal:')
        print(torch.isclose(model(random_input), model2(random_input), atol=1e-03).all())
        #print(model.layer4[2].conv3[0])
        #print(model.layer4[2].conv3[0].weight.shape) 
    
    return model, layer_path


class SAEDataset(torch.utils.data.Dataset):
    """
    Memory-efficient dataset implementation for Sparse Autoencoders.
    
    This class loads data from memory-mapped numpy arrays to efficiently handle
    large datasets without loading everything into memory at once. It also
    handles preprocessing like mean centering and normalization.
    
    The class automatically parses dataset dimensions from the filename,
    which is expected to contain the data shape as the last two underscored
    components (e.g., "dataset_name_10000_768.npy" for 10000 vectors of size 768).
    
    Args:
        data_path (str): Path to the memory-mapped numpy array file
        dtype (torch.dtype, optional): Data type for tensors. Defaults to torch.float32.
        mean_center (bool, optional): Whether to center the data by subtracting the mean.
                                     Defaults to False.
        target_norm (float, optional): Target norm for normalization. If None, uses sqrt(vector_size).
                                     If 0.0, no normalization is applied. Defaults to None.
    """
    def __init__(self, data_path: str, dtype: torch.dtype = torch.float32, mean_center: bool = False, target_norm: float = None, split="train", model_type="vit_b_16"):
        # Parse vector dimensions from filename
        self.len = 50000 if split == "val" else 1281167
        if model_type == "resnet50":
            self.vector_size = 2048 
            if "input" in data_path:
                self.vector_size = 512
        elif model_type == "dinov2_vitb14":
            self.vector_size = 768
            if "input" in data_path:
                self.vector_size = 3072
        elif model_type == "vit_b_16":
            self.vector_size = 768
            if "input" in data_path:
                self.vector_size = 3072
        else:
            self.vector_size = 1024
            if "input" in data_path:
                self.vector_size = 4096
        #parts = data_path.split("/")[-1].split(".")[0].split("_")
        #self.len, self.vector_size = map(int, parts[-2:])
        
        # Set core attributes
        #print((self.len, self.vector_size))
        self.dtype = dtype
        self.data = np.memmap(data_path, dtype="float32", mode="r", 
                             shape=(self.len, self.vector_size))
        
        # Special case for representation files (already preprocessed)
        if "repr" in data_path:
            self.mean = torch.zeros(self.vector_size, dtype=dtype)
            self.mean_center = False
            self.scaling_factor = 1.0
            return

        # Set preprocessing configuration
        self.mean_center = mean_center
        self.target_norm = np.sqrt(self.vector_size) if target_norm is None else target_norm

        # Compute statistics if needed
        if self.mean_center or self.target_norm != 0.0:
            self._compute_statistics()
        else:
            self.mean = torch.zeros(self.vector_size, dtype=dtype)
            self.scaling_factor = 1.0

    def _compute_statistics(self, batch_size: int = 10000):
        """
        Compute dataset statistics (mean and scaling factor) in memory-efficient batches.
        
        Args:
            batch_size (int, optional): Number of samples to process at once. Defaults to 10000.
        """
        # Compute mean if mean centering is enabled
        if self.mean_center:
            mean_acc = np.zeros(self.vector_size, dtype=np.float32)
            total = 0

            for start in range(0, self.len, batch_size):
                end = min(start + batch_size, self.len)
                batch = self.data[start:end].copy()
                mean_acc += np.sum(batch, axis=0)
                total += (end - start)

            self.mean = torch.from_numpy(mean_acc / total).to(self.dtype)
        else:
            self.mean = torch.zeros(self.vector_size, dtype=self.dtype)

        # Compute scaling factor if normalization is enabled
        if self.target_norm != 0.0:
            squared_norm_sum = 0.0
            total = 0

            for start in range(0, self.len, batch_size):
                end = min(start + batch_size, self.len)
                batch = self.data[start:end].copy()
                # Center the batch if needed
                batch = batch - self.mean.numpy()
                squared_norm_sum += np.sum(np.square(batch))
                total += (end - start)

            avg_squared_norm = squared_norm_sum / total
            self.scaling_factor = float(self.target_norm / np.sqrt(avg_squared_norm))
        else:
            self.scaling_factor = 1.0

    def __len__(self):
        """Return the number of samples in the dataset."""
        return self.len
    
    def process_data(self, data: torch.Tensor) -> torch.Tensor:
        """
        Process data for the autoencoder (subtract mean and apply scaling).
        
        Args:
            data (torch.Tensor): Input data tensor
            
        Returns:
            torch.Tensor: Processed data tensor
        """        
        data.sub_(self.mean)
        data.mul_(self.scaling_factor)
        
        return data
    
    def unprocess_data(self, data: torch.Tensor) -> torch.Tensor:
        """
        Reverse the processing of data (apply inverse scaling and add mean).
        
        Args:
            data (torch.Tensor): Input data tensor
            
        Returns:
            torch.Tensor: Unprocessed data tensor
        """        
        data.div_(self.scaling_factor)
        data.add_(self.mean)
        
        return data

    @torch.no_grad()
    def __getitem__(self, idx):
        """
        Get a preprocessed data sample at the specified index.
        
        Args:
            idx (int): Index of the sample to retrieve
            
        Returns:
            torch.Tensor: Preprocessed data sample
        """
        torch_data = torch.tensor(self.data[idx])
        output = self.process_data(torch_data.clone())
        return output.to(self.dtype)


class LinearDecayLR(torch.optim.lr_scheduler.LambdaLR):
    """
    Learning rate scheduler with a constant phase followed by linear decay.
    
    The learning rate remains constant for a specified fraction of total epochs,
    then decays linearly to zero for the remaining epochs.
    
    Args:
        optimizer (torch.optim.Optimizer): The optimizer to adjust
        total_epochs (int): Total number of training epochs
        decay_time (float, optional): Fraction of total epochs before decay starts.
                                     Defaults to 0.8 (80% of training).
        last_epoch (int, optional): The index of the last epoch. Defaults to -1.
    """
    def __init__(self, optimizer, total_epochs, decay_time = 0.8, last_epoch=-1):
        def lr_lambda(epoch):
            if epoch < int(decay_time * total_epochs):
                return 1.0
            return max(0.0, (total_epochs - epoch) / ((1-decay_time) * total_epochs))
        
        super().__init__(optimizer, lr_lambda, last_epoch)


class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """
    Learning rate scheduler with warmup and cosine annealing.
    
    This scheduler implements:
    1. Linear warmup from initial_lr (max_lr * final_lr_factor) to max_lr during the warmup epoch
    2. Cosine annealing from max_lr to final_lr (max_lr * final_lr_factor) for the remaining epochs
    
    Args:
        optimizer (torch.optim.Optimizer): The optimizer to adjust
        max_lr (float): Maximum learning rate after warmup
        total_epochs (int): Total number of training epochs
        warmup_epoch (int, optional): Number of warmup epochs. Defaults to 1.
        final_lr_factor (float, optional): Ratio of final LR to max LR. Defaults to 0.1.
        last_epoch (int, optional): The index of the last epoch. Defaults to -1.
    """
    def __init__(self, optimizer, max_lr, total_epochs, warmup_epoch=1, 
                 final_lr_factor=0.1, last_epoch=-1):
        self.max_lr = max_lr
        self.total_epochs = total_epochs
        self.warmup_epoch = warmup_epoch
        self.initial_lr = max_lr * final_lr_factor
        self.final_lr = max_lr * final_lr_factor
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """
        Calculate the learning rate for the current epoch.
        
        Returns:
            list: Learning rates for each parameter group
        """
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                        "please use `get_last_lr()`.")

        # During warmup (first epoch)
        if self.last_epoch < self.warmup_epoch:
            # Linear interpolation from initial_lr to max_lr
            alpha = self.last_epoch / self.warmup_epoch
            return [self.initial_lr + (self.max_lr - self.initial_lr) * alpha 
                    for _ in self.base_lrs]
        
        # After warmup - Cosine annealing
        else:
            # Adjust epoch count to start cosine annealing after warmup
            current = self.last_epoch - self.warmup_epoch
            total = self.total_epochs - self.warmup_epoch
            
            # Implement cosine annealing
            cosine_factor = (1 + math.cos(math.pi * current / total)) / 2
            return [self.final_lr + (self.max_lr - self.final_lr) * cosine_factor 
                    for _ in self.base_lrs]


def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility across all random number generators.
    
    Args:
        seed (int): The seed value to use
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """
    Determine the best available device for PyTorch computation.
    
    Returns:
        torch.device: The selected device (CUDA if available, MPS on Apple Silicon, CPU otherwise)
    """
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    return torch.device(device)


def normalize_data(x: torch.Tensor, eps: float = 1e-5) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Normalize input data to zero mean and unit variance.
    
    Args:
        x (torch.Tensor): Input tensor to normalize
        eps (float, optional): Small constant for numerical stability. Defaults to 1e-5.
        
    Returns:
        tuple: (normalized_data, mean, std)
            - normalized_data: Data normalized to zero mean and unit variance
            - mean: Mean of the original data (for denormalization)
            - std: Standard deviation of the original data (for denormalization)
    """
    mu = x.mean(dim=-1, keepdim=True)
    x = x - mu
    std = x.std(dim=-1, keepdim=True)
    x = x / (std + eps)
    return x, mu, std


@torch.no_grad()
def geometric_median(dataset: torch.utils.data.Dataset, eps: float = 1e-5, 
                    device: torch.device = torch.device("cpu"), 
                    max_number: int = 925117, max_iter: int = 1000) -> torch.Tensor:
    """
    Compute the geometric median of a dataset using Weiszfeld's algorithm.
    
    The geometric median is a generalization of the median to multiple dimensions
    and is robust to outliers. This implementation uses iterative approximation
    with early stopping based on convergence.
    
    Args:
        dataset (torch.utils.data.Dataset): The dataset to compute median for
        eps (float, optional): Convergence threshold. Defaults to 1e-5.
        device (torch.device, optional): Computation device. Defaults to CPU.
        max_number (int, optional): Maximum number of samples to use. Defaults to 925117.
        max_iter (int, optional): Maximum number of iterations. Defaults to 1000.
        
    Returns:
        torch.Tensor: The geometric median vector
    """
    # Sample a subset of the dataset if it's large
    indices = torch.randperm(len(dataset))[:min(len(dataset), max_number)]
    X = dataset[indices]
    
    # Move data to device
    try:
        X = X.to(device)
    except Exception as e:
        warnings.warn(f"Error moving dataset to device: {device}, using default device {X.device}")
    
    # Initialize with arithmetic mean
    y = torch.mean(X, dim=0)
    progress_bar = tqdm(range(max_iter), desc="Geometric Median Iteration", leave=False)
    
    # Weiszfeld's algorithm
    for _ in progress_bar:
        # Compute distances to current estimate
        D = torch.norm(X - y, dim=1)
        nonzeros = (D != 0)  # Avoid division by zero
        
        # Compute weights for non-zero distances
        Dinv = 1 / D[nonzeros]
        Dinv_sum = torch.sum(Dinv)
        W = Dinv / Dinv_sum
        
        # Weighted average of points
        T = torch.sum(W.view(-1, 1) * X[nonzeros], dim=0)
        
        # Handle special case when some points equal the current estimate
        num_zeros = len(X) - torch.sum(nonzeros)
        if num_zeros == 0:
            # No points equal the current estimate
            y1 = T
        else:
            # Some points equal the current estimate
            R = T * Dinv_sum / (Dinv_sum - num_zeros)
            r = torch.norm(R - y)
            progress_bar.set_postfix({"r": r.item()})
            if r < eps:
                return y
            y1 = R
        
        # Check convergence
        if torch.norm(y - y1) < eps:
            return y1
        
        y = y1
    
    # Return best estimate after max iterations
    return y


def calculate_vector_mean(dataset: torch.utils.data.Dataset,
                          batch_size: int = 10000,
                          num_workers: int = 4) -> torch.Tensor:
    """
    Efficiently calculate the mean of vectors in a dataset.
    
    This function processes the dataset in batches to handle large datasets
    that might not fit in memory all at once.
    
    Args:
        dataset (torch.utils.data.Dataset): Dataset containing vectors
        batch_size (int, optional): Batch size for processing. Defaults to 10000.
        num_workers (int, optional): Number of worker processes for data loading. Defaults to 4.
        
    Returns:
        torch.Tensor: Mean vector of the dataset
    """
    # Use DataLoader to efficiently iterate through the dataset
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False  # No need to shuffle for calculating mean
    )
    
    # Initialize sum and count
    vector_sum = torch.zeros_like(dataset[0])
    count = 0
    
    # Iterate through batches
    for batch in tqdm(dataloader, desc="Calculating Mean Vector", leave=False):
        batch_count = batch.size(0)
        vector_sum += batch.sum(dim=0)
        count += batch_count
    
    # Calculate mean
    mean_vector = vector_sum / count
    
    return mean_vector


class RectangleFunction(torch.autograd.Function):
    """
    Custom autograd function that implements a rectangle function.
    
    This function outputs 1.0 for inputs between -0.5 and 0.5, and 0.0 elsewhere.
    The gradient is non-zero only within this interval.
    
    Used as a building block for other activation functions with custom gradients.
    """
    @staticmethod
    def forward(ctx, x):
        """
        Forward pass of the rectangle function.
        
        Args:
            ctx: Context for saving variables for backward
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output tensor with values in {0.0, 1.0}
        """
        ctx.save_for_backward(x)
        return ((x > -0.5) & (x < 0.5)).float()

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass of the rectangle function.
        
        Args:
            ctx: Context with saved variables
            grad_output (torch.Tensor): Gradient from subsequent layers
            
        Returns:
            torch.Tensor: Gradient with respect to input
        """
        (x,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[(x <= -0.5) | (x >= 0.5)] = 0
        return grad_input


class JumpReLUFunction(torch.autograd.Function):
    """
    Custom autograd function implementing a thresholded ReLU with learnable threshold.
    
    This activation function passes values through only if they exceed a learned threshold.
    It has custom gradients for both the input and the threshold parameter.
    """
    @staticmethod
    def forward(ctx, x, log_threshold, bandwidth):
        """
        Forward pass of the JumpReLU function.
        
        Args:
            ctx: Context for saving variables for backward
            x (torch.Tensor): Input tensor
            log_threshold (torch.Tensor): Log of the threshold value (learned parameter)
            bandwidth (float): Bandwidth parameter for gradient approximation
            
        Returns:
            torch.Tensor: Output tensor
        """
        ctx.save_for_backward(x, log_threshold, torch.tensor(bandwidth))
        threshold = torch.exp(log_threshold)
        return x * (x > threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass of the JumpReLU function.
        
        Args:
            ctx: Context with saved variables
            grad_output (torch.Tensor): Gradient from subsequent layers
            
        Returns:
            tuple: (input_gradient, threshold_gradient, None)
        """
        x, log_threshold, bandwidth_tensor = ctx.saved_tensors
        bandwidth = bandwidth_tensor.item()
        threshold = torch.exp(log_threshold)
        
        # Gradient with respect to x
        x_grad = (x > threshold).float() * grad_output
        
        # Gradient with respect to threshold
        # Uses rectangle function to approximate the dirac delta
        threshold_grad = (
            -(threshold / bandwidth)
            * RectangleFunction.apply((x - threshold) / bandwidth)
            * grad_output
        )
        
        return x_grad, threshold_grad, None  # None for bandwidth


class StepFunction(torch.autograd.Function):
    """
    Custom autograd function implementing a step function with learnable threshold.
    
    This activation function outputs 1 for values above a threshold and 0 otherwise.
    It has custom gradients for both the input and the threshold parameter.
    """
    @staticmethod
    def forward(ctx, x, log_threshold, bandwidth):
        """
        Forward pass of the step function.
        
        Args:
            ctx: Context for saving variables for backward
            x (torch.Tensor): Input tensor
            log_threshold (torch.Tensor): Log of the threshold value (learned parameter)
            bandwidth (float): Bandwidth parameter for gradient approximation
            
        Returns:
            torch.Tensor: Binary output tensor with values in {0.0, 1.0}
        """
        ctx.save_for_backward(x, log_threshold, torch.tensor(bandwidth))
        threshold = torch.exp(log_threshold)
        return (x > threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass of the step function.
        
        Args:
            ctx: Context with saved variables
            grad_output (torch.Tensor): Gradient from subsequent layers
            
        Returns:
            tuple: (input_gradient, threshold_gradient, None)
        """
        x, log_threshold, bandwidth_tensor = ctx.saved_tensors
        bandwidth = bandwidth_tensor.item()
        threshold = torch.exp(log_threshold)
        
        # No gradient with respect to x (step function)
        x_grad = torch.zeros_like(x)
        
        # Gradient with respect to threshold
        # Uses rectangle function to approximate the dirac delta
        threshold_grad = (
            -(1.0 / bandwidth)
            * RectangleFunction.apply((x - threshold) / bandwidth)
            * grad_output
        )
        
        return x_grad, threshold_grad, None  # None for bandwidth
