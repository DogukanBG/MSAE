import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from FMS_IN1k import *

import os
import torch
import argparse
import logging
import numpy as np
from tqdm import tqdm
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder, CelebA
import torch.nn as nn
from imagenet1000_class_names import imagenet1000_class_names

import torch
import logging
import argparse
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score

from sae import load_model
from utils import SAEDataset, set_seed, get_device
import utils as ut

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

# ============================================================================
# FMS (Feature Manipulation Score)
# ============================================================================

def load_tree_stats_data(file_info_list) -> pd.DataFrame:
    """Load and combine tree statistics for FMS computation."""
    all_data = pd.DataFrame()
    
    for info in file_info_list:
        if "cut" in info["file"]:
            continue
        
        df = pd.read_csv(info["file"])
        df["model_name"] = info["model_name"]
        df["concept"] = info["concept"]
        df["model type"] = info["model_type"]
        
        all_data = pd.concat([all_data, df], ignore_index=True)
    
    # Compute global model shift
    all_data["MS_global"] = all_data.apply(
        lambda x: 1 - (
            (all_data[
                (all_data["Nodes"] != 1) & 
                (all_data["model type"] == x["model type"]) & 
                (all_data["concept"] == x["concept"])
            ]["Accuracy"] - all_data[
                (all_data["Nodes"] == 1) & 
                (all_data["model type"] == x["model type"]) & 
                (all_data["concept"] == x["concept"])
            ]["Accuracy"].item()).sum() / 
            len(all_data[
                (all_data["Nodes"] != 1) & 
                (all_data["model type"] == x["model type"]) & 
                (all_data["concept"] == x["concept"])
            ])
        ) if x["Nodes"] == 1 else None,
        axis=1
    )
    
    return all_data[all_data["Nodes"] == 1][["Accuracy", "concept", "model type", "MS_global"]]


def load_local_tree_stats_data(file_info_list) -> pd.DataFrame:
    """Load local tree statistics for FMS computation."""
    all_data = pd.DataFrame()
    
    for info in file_info_list:
        df = pd.read_csv(info["file"])
        df["model_name"] = info["model_name"]
        df["concept"] = info["concept"]
        df["model type"] = info["model_type"]
        
        df["Accuracy"] = pd.to_numeric(df["Accuracy"], errors="coerce")
        
        # Compute local model shift
        df["MS_local"] = df.apply(
            lambda x: 2 * (
                df[(df["Nodes"] == 1) & (df["num_cuts"] == 0)]["Accuracy"].mean() - x["Accuracy"]
            ) if x["num_cuts"] != 0 else None,
            axis=1
        )
        
        all_data = pd.concat([all_data, df], ignore_index=True)
    
    return all_data[all_data["Nodes"] == 1][["num_cuts", "concept", "model type", "MS_local"]]


def compute_fms_score(model, sae, experiment_name, layer_path="layer4.2.conv3", 
                      file_path="/BS/disentanglement/work/fms", num_pairs=100) -> float:
    """
    Compute Feature Manipulation Score (FMS).
    Tests model behavior under feature manipulations using decision trees.
    """
    import random
    from low_sim_pairs import LOW_SIM_PAIRS
    rng = random.Random(42)
    
    #Generate random class pairs (sufficiently different)
    pairs = []
    while len(pairs) < num_pairs:
       x = rng.randint(0, 999)
       y = rng.randint(0, 999)
       if abs(x - y) >= 100:
           pairs.append((x, y))
            
    #pairs = [(721, 299), (422, 744), (941, 703), (420, 613), (406, 995), (585, 442), (769, 917), (119, 226), (973, 91), (234, 926), (659, 164), (441, 313), (672, 321), (405, 9), (45, 489), (41, 149), (504, 288), (32, 384), (575, 750), (756, 329), (625, 395), (848, 387), (451, 567), (955, 67), (217, 934), (851, 697), (912, 115), (416, 255), (775, 661), (330, 156), (286, 773), (522, 917), (509, 953), (806, 465), (646, 210), (649, 41), (880, 537), (666, 421), (955, 158), (640, 349), (366, 76), (163, 45), (875, 73), (195, 977), (451, 178), (751, 392), (196, 925), (323, 459), (967, 257), (992, 447), (951, 278), (374, 199), (120, 661), (49, 375), (161, 855), (828, 508), (907, 127), (379, 590), (249, 427), (488, 145), (944, 531), (404, 525), (568, 182), (610, 822), (742, 230), (928, 494), (444, 780), (947, 560), (603, 721), (208, 49), (999, 800), (735, 282), (652, 892), (92, 616), (366, 246), (209, 982), (115, 238), (379, 64), (478, 854), (177, 592), (949, 403), (755, 288), (748, 86), (265, 551), (922, 713), (97, 786), (115, 339), (607, 241), (362, 232), (618, 922), (130, 26), (488, 74), (627, 828), (14, 219), (731, 998), (812, 246), (77, 336), (544, 412), (72, 654), (564, 438)]
    fms_scores = []
    
    for class1, class2 in tqdm(pairs, desc="Computing FMS"):
        # Get subdatasets for the two classes
        dataset1 = get_subdataset_for_classes(
            dir="/scratch/inf0/user/mparcham/ILSVRC2012/val_categorized",
            class_idx_list=[class1]
        )
        dataloader1 = torch.utils.data.DataLoader(dataset1, batch_size=32)
        
        dataset2 = get_subdataset_for_classes(
            dir="/scratch/inf0/user/mparcham/ILSVRC2012/val_categorized",
            class_idx_list=[class2]
        )
        dataloader2 = torch.utils.data.DataLoader(dataset2, batch_size=32)
        
        # Run tree analysis
        os.makedirs(f"{file_path}/{experiment_name}", exist_ok=True)
        #experiment_name = experiment_name + f"{class1}-{class2}"
        IN1k_tree(experiment_name, model, sae, layer_path, dataloader1, dataloader2, file_path, plot=False)
        cut_IN1k_tree(experiment_name, model, sae, layer_path, dataloader1, dataloader2, file_path, label_shuffle=False)
        
        # Load and merge results
        file_info = [{
            "file": f"{file_path}/{experiment_name}_stats.csv",
            "model_name": experiment_name,
            "concept": "ImageNet",
            "model_type": "SAE"
        }]
        df_global = load_tree_stats_data(file_info)
        
        file_info[0]["file"] = f"{file_path}/{experiment_name}_cut.csv"
        df_local = load_local_tree_stats_data(file_info)
        
        # Compute FMS
        df = pd.merge(df_local, df_global)
        df["FMS"] = df.apply(lambda x: x["Accuracy"] * ((x["MS_local"] + x["MS_global"]) / 2), axis=1)
        
        fms_score = df.loc[df["num_cuts"] == 5, "FMS"].iloc[0]
        fms_scores.append(fms_score)
    
    return np.mean(fms_scores)

def create_data_loaders(root="/scratch/inf0/user/mparcham/ILSVRC2012/val_categorized", batch_size: int = 256, num_workers: int = 4, image_size: int = 224):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    dataset = ImageFolder(root=root, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    return loader, len(dataset)


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments for the linear probe evaluation.
    
    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description="Evaluate semantic preservation in SAE reconstructions using a linear probe")
    parser.add_argument("-m", "--model", type=str, required=True, 
                       help="Path to the trained SAE model file")
    parser.add_argument("--sae", type=str, 
                       help="Path to the trained SAE model file")
    parser.add_argument("-b", "--batch-size", type=int, default=512, 
                       help="Batch size for training and evaluation")
    parser.add_argument("-s", "--seed", type=int, default=42, 
                       help="Random seed for reproducibility")
    parser.add_argument("-w", "--num-workers", type=int, default=8, 
                       help="Number of worker processes for data loading")
    parser.add_argument("--eval-hc", type=bool, default=False, 
                       help="Number of worker processes for data loading")
    parser.add_argument("-t", "--tau", type=float, default=0.03, 
                       help="Batch size for training and evaluation")

    return parser.parse_args()


def get_model(args, model_name: str, device: str):
    """
    Returns a model for image embeddings.
    """
    if model_name.lower() == "resnet50":
        model = torchvision.models.resnet50(pretrained=True)
        #model = nn.Sequential(*list(model.children())[:-1])  # Remove classifier
        embedding_dim = 2048 
        layer_path="layer4.2.conv3" if not args.eval_hc else "layer4.2.conv3.0"
    elif model_name.lower() == "convnext_tiny":
        model = torchvision.models.convnext_tiny(pretrained=True)
        #model = nn.Sequential(*list(model.features), nn.AdaptiveAvgPool2d(1))
        embedding_dim = 768
    elif model_name.lower() == "vit_b_16":
        model = torchvision.models.vit_b_16(pretrained=True)
        #model.heads = nn.Identity()
        embedding_dim = 768
        layer_path = "encoder.layers.encoder_layer_11.mlp.3" if not args.eval_hc else "encoder.layers.encoder_layer_11.mlp.3.0"
    elif model_name.lower() == "dinov2_vitl14":
        import torch.hub
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
        embedding_dim =1024
        layer_path="blocks.23.mlp.fc2" if not args.eval_hc else "blocks.23.mlp.fc2.0"
    else:
        raise ValueError(f"Unsupported model {model_name}")
    
    model.to(device).eval()
    return model, layer_path, embedding_dim

def main(args):
    #model
    #sae
    #experiment_name
    #layer_path
        # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Get the device to use for training
    device = get_device()
    model, layer_path, embedding_dim = get_model(args, args.model, device)
    model2, layer_path, embedding_dim = get_model(args, args.model, device)
    experiment_name = args.model
    if args.eval_hc:
        print("Chosen model", args.model)
        if args.model == "resnet50":
            model, _ = ut.load_fully_multi_disentangled_resnet(tau=args.tau)
        elif args.model == "vit_b_16":
            model, _ = ut.load_fully_multi_disentangled_vit(tau=args.tau)
        elif args.model == "dinov2_vitl14":
            model, _ = ut.load_fully_multi_disentangled_dinov2(model2, tau=args.tau)
        sae=None
    else:
        # Load the trained SAE model
        sae, mean_center, scaling_factor, target_norm = load_model(args.sae)        
        sae.to(device)
        sae.eval()
    
    #model, layer_path, embedding_dim = get_model(args, args.model, device)
    fms = compute_fms_score(model, sae, experiment_name, layer_path)
    
    experiment_name = f"hc{args.tau}" + args.model if args.eval_hc else args.sae
    results = {
            "Model Name": [experiment_name],
            "FMS": [fms],
        }
    print(results)
    df = pd.DataFrame(results)

    # Append instead of overwrite
    csv_file = "/BS/disentanglement/work/Disentanglement/MSAE/results_fms.csv"
    df.to_csv(csv_file, mode="a", index=False, header=not os.path.exists(csv_file))
    return 

if __name__ == "__main__":
    args = parse_args()
    main(args)