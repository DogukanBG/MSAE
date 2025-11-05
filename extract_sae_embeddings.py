import os
import torch
import logging
import argparse
import numpy as np
from tqdm import tqdm
import random

from sae import load_model
from utils import SAEDataset, set_seed, get_device
from metrics import (
    explained_variance_full,
    normalized_mean_absolute_error,
    l0_messure,
    cknna,
    orthogonal_decoder
)
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

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

def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments for the representation extraction and evaluation script.
    
    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description="Extract and evaluate representations from Sparse Autoencoder models")
    parser.add_argument("-m", "--model", type=str, required=True, 
                        help="Path to the trained model file (.pt)")
    parser.add_argument("-d", "--data", type=str, required=True, 
                        help="Path to the dataset file (.npy)")
    parser.add_argument("-b", "--batch-size", type=int, default=10000, 
                        help="Batch size for processing data")
    parser.add_argument("-o", "--output-path", type=str, default=".", 
                        help="Directory path to save extracted representations")
    parser.add_argument("-s", "--seed", type=int, default=42, 
                        help="Random seed for reproducibility")

    return parser.parse_args()


def test_size(model) -> float:
    """Return total number of parameters in millions."""
    print("Computing model size...")
    total_params = sum(p.numel() for p in model.parameters())
    return round(total_params / 1_000_000, 1)


def get_representation(model_path_name, model, dataset, repr_file_name, batch_size):
    """
    Extract representations from the model for the given dataset and evaluate model performance.
    
    Extracts both output reconstructions and latent representations from the model,
    saves them to disk as memory-mapped files, and computes various performance metrics.
    
    Args:
        model: The Sparse Autoencoder model to evaluate
        dataset: Dataset to process
        repr_file_name (str): Base filename for saving representations
        batch_size (int): Number of samples to process at once
        
    Metrics computed:
        - Fraction of Variance Unexplained (FVU) using normalized MSE
        - Normalized Mean Absolute Error (MAE)
        - Cosine similarity between inputs and outputs
        - L0 measure (average number of active neurons per sample)
        - CKNNA (Cumulative k-Nearest Neighbor Accuracy)
        - Number of dead neurons (neurons that never activate)
    """
    device = get_device()
    logger.info(f"Using device: {device}")
    
    model.eval()
    model.to(device)
    with torch.no_grad():
        # Prepare memory-mapped file for output reconstructions
        repr_file_name_output = f"{repr_file_name}_output_{len(dataset)}_{model.n_inputs}.npy"
        memmap_output = np.memmap(repr_file_name_output, dtype='float32', mode='w+', 
                                  shape=(len(dataset), model.n_inputs))
        logger.info(f"Data output with shape {memmap_output.shape} will be saved to {repr_file_name_output}")

        # Prepare memory-mapped file for latent representations
        repr_file_name_repr = f"{repr_file_name}_repr_{len(dataset)}_{model.n_latents}.npy"
        memmap_repr = np.memmap(repr_file_name_repr, dtype='float32', mode='w+', 
                                shape=(len(dataset), model.n_latents))
        logger.info(f"Data repr with shape {memmap_repr.shape} will be saved to {repr_file_name_repr}")

        # Create dataloader for batch processing
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                               shuffle=True, num_workers=0)
        
        # Lists to collect metrics for each batch
        l0 = []
        mae = []
        fvu = []
        cs = []
        cknnas = []
        sparse_l0 = []
        sparse_mae = []
        sparse_fvu = []
        sparse_cs = []
        sparse_cknnas = []
        dead_neurons_count = None
        
        size = test_size(model) #+ test_size(model.encoder)
        do = orthogonal_decoder(model.decoder)
        results = {
           "Model Name": [model_path_name],
           "DO": [do]
        }
        print(results)
        return 
        
        #df = pd.DataFrame(results)

        # Append instead of overwrite
        #csv_file = "/BS/disentanglement/work/Disentanglement/MSAE/results_do.csv"
        #df.to_csv(csv_file, mode="a", index=False, header=not os.path.exists(csv_file))
        
        #print(results)

        # Process data in batches
        for idx, batch in enumerate(tqdm(dataloader, desc="Extracting representations")):
            start = batch_size * idx
            end = start + batch.shape[0]
            batch = batch.to(device)
            
            # Forward pass through the model
            with torch.no_grad():
                sparse_outputs, sparse_representation, outputs, representations = model(batch)
            
            # Post-process outputs and batch
            batch = dataset.unprocess_data(batch.cpu()).to(device)
            outputs = dataset.unprocess_data(outputs.cpu()).to(device)
            sparse_outputs = dataset.unprocess_data(sparse_outputs.cpu()).to(device)
            
            
            # Save the outputs and representations to the memmap files
            outputs_numpy = outputs.cpu().numpy()
            memmap_output[start:end] = outputs_numpy
            memmap_output.flush()

            representations_numpy = representations.cpu().numpy()
            memmap_repr[start:end] = representations_numpy
            memmap_repr.flush()

            # Calculate and collect metrics
            fvu.append(explained_variance_full(batch, outputs))
            mae.append(normalized_mean_absolute_error(batch, outputs))
            cs.append(torch.nn.functional.cosine_similarity(batch, outputs))
            l0.append(l0_messure(representations))
            # Only calculate the cknna if it even to the number of the batch
            if batch.shape[0] == batch_size:
                cknnas.append(cknna(batch, representations, topk=10))
            
            sparse_fvu.append(explained_variance_full(batch, sparse_outputs))
            sparse_mae.append(normalized_mean_absolute_error(batch, sparse_outputs))
            sparse_cs.append(torch.nn.functional.cosine_similarity(batch, sparse_outputs))
            sparse_l0.append(l0_messure(sparse_representation))
            # Only calculate the cknna if it even to the number of the batch
            if batch.shape[0] == batch_size:
                sparse_cknnas.append(cknna(batch, sparse_representation, topk=10))
            
            # Track neurons that are activated at least once
            if dead_neurons_count is None:
                dead_neurons_count = (representations != 0).sum(dim=0).cpu().long()
            else:
                dead_neurons_count += (representations != 0).sum(dim=0).cpu().long()

        # Aggregate metrics across all batches
        mae = torch.cat(mae, dim=0).cpu().numpy()
        cs = torch.cat(cs, dim=0).cpu().numpy()
        l0 = torch.cat(l0, dim=0).cpu().numpy()
        fvu = torch.cat(fvu, dim=0).cpu().numpy()
        cknnas = np.array(cknnas)
        sparse_mae = torch.cat(sparse_mae, dim=0).cpu().numpy()
        sparse_cs = torch.cat(sparse_cs, dim=0).cpu().numpy()
        sparse_l0 = torch.cat(sparse_l0, dim=0).cpu().numpy()
        sparse_fvu = torch.cat(sparse_fvu, dim=0).cpu().numpy()
        sparse_cknnas = np.array(sparse_cknnas)
        
        # Count neurons that were never activated
        number_of_dead_neurons = torch.where(dead_neurons_count == 0)[0].shape[0]

        # Log final metrics
        logger.info(f"Fraction of Variance Unexplained (FVU): {np.mean(fvu)} +/- {np.std(fvu)}")
        logger.info(f"Normalized MAE: {np.mean(mae)} +/- {np.std(mae)}")
        logger.info(f"Cosine similarity: {np.mean(cs)} +/- {np.std(cs)}")
        logger.info(f"L0 messure: {np.mean(l0)} +/- {np.std(l0)}")
        logger.info(f"CKNNA: {np.mean(cknnas)} +/- {np.std(cknnas)}")
        logger.info(f"Number of dead neurons: {number_of_dead_neurons}")
        logger.info(f"\nSparse Fraction of Variance Unexplained (FVU): {np.mean(sparse_fvu)} +/- {np.std(sparse_fvu)}")
        logger.info(f"Sparse Normalized MAE: {np.mean(sparse_mae)} +/- {np.std(sparse_mae)}")
        logger.info(f"Sparse Cosine similarity: {np.mean(sparse_cs)} +/- {np.std(sparse_cs)}")
        logger.info(f"Sparse L0 measure: {np.mean(sparse_l0)} +/- {np.std(sparse_l0)}")
        logger.info(f"Sparse CKNNA: {np.mean(sparse_cknnas)} +/- {np.std(sparse_cknnas)}")
        
        results = {
            "Model Name": [model_path_name],
            "Number of dead neurons": [number_of_dead_neurons],
            "Fraction of Variance Unexplained (FVU)": [np.mean(sparse_fvu)],
            "Normalized MAE": [np.mean(sparse_mae)],
            "Cosine similarity": [np.mean(sparse_cs)],
            "L0 measure": [np.mean(sparse_l0)],
            "CKNNA": [np.mean(sparse_cknnas)],
            #"DO": [do],
            "Size": [size]
        }

        df = pd.DataFrame(results)

        # Append instead of overwrite
        csv_file = "/BS/disentanglement/work/Disentanglement/MSAE/results_sae.csv"
        df.to_csv(csv_file, mode="a", index=False, header=not os.path.exists(csv_file))

import torchvision
def get_model(model_name: str, device: str):
    """
    Returns a model for image embeddings.
    """
    if model_name.lower() == "resnet50":
        model = torchvision.models.resnet50(pretrained=True)
        #model = nn.Sequential(*list(model.children())[:-1])  # Remove classifier
        embedding_dim = 2048 
        layer_path="layer4.2.conv3" 
    elif model_name.lower() == "convnext_tiny":
        model = torchvision.models.convnext_tiny(pretrained=True)
        #model = nn.Sequential(*list(model.features), nn.AdaptiveAvgPool2d(1))
        embedding_dim = 768
    elif model_name.lower() == "vit_b_16":
        model = torchvision.models.vit_b_16(pretrained=True)
        #model.heads = nn.Identity()
        embedding_dim = 768
        layer_path = "encoder.layers.encoder_layer_11.mlp.3" 
    elif model_name.lower() == "dinov2_vitl14":
        import torch.hub
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
        embedding_dim =1024
        layer_path="blocks.23.mlp.fc2" 
    else:
        raise ValueError(f"Unsupported model {model_name}")
    
    model.to(device).eval()
    return model, layer_path, embedding_dim

import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

def create_data_loaders(root: str, batch_size: int = 256, num_workers: int = 0, image_size: int = 224):    
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

def main(args):
    """
    Main function to load model and dataset, then extract and evaluate representations.
    
    Args:
        args (argparse.Namespace): Command line arguments.
    """
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Load the trained model
    model, mean_center, scaling_factor, target_norm = load_model(args.model)
    logger.info("Model loaded")
    
    model_type = None
    if "resnet50" in args.data:
        model_type = "resnet50"
    elif "vit_b_16" in args.data:
        model_type = "vit_b_16"
    elif "dinov2_vitl14" in args.data:
        model_type = "dinov2_vitl14"
        
    model_orig, layer_path, embedding_dim = get_model(model_name=model_type, device="cuda:0")
    
    # Load the dataset with appropriate preprocessing
    if ("text" in args.model and "text" in args.data) or ("image" in args.model and "image" in args.data):
        logger.info("Using model mean and scalling factor")    
        dataset = SAEDataset(args.data, split="val", model_type=model_type)
        dataset.mean = mean_center.cpu()
        dataset.scaling_factor = scaling_factor
    else:    
        logger.info("Computing mean and scalling factor")    
        dataset = SAEDataset(args.data, mean_center=True if mean_center.sum() != 0.0 else False, target_norm=target_norm, split="val", model_type=model_type)
        
    logger.info(f"Dataset loaded with length: {len(dataset)}")
    logger.info(f"Dataset mean center: {dataset.mean.mean()}, Scaling factor: {dataset.scaling_factor} with target norm {dataset.target_norm}")
    # Construct output filename from model and data names
    model_path_name = args.model.split("/")[-1].replace(".pt","") if not ("batchtopk" in args.model) else args.model.split("/")[-1].replace(".pt","") + "batchtopk"
    data_path_name = args.data.split("/")[-1].replace(".npy","")
    repr_file_name = os.path.join(args.output_path, f"{data_path_name}_{model_path_name}")
    
    # Extract representations and compute metrics
    #get_representation(model_path_name, model, dataset, repr_file_name, args.batch_size)

    dataloader, _ = create_data_loaders("/scratch/inf0/user/mparcham/ILSVRC2012/val_categorized", num_workers=0)
    import metrics as m
    
    # Example usage:

    # Run the metric (first time - will compute and cache)
    experiment_name = model_path_name
    metrics = m.compute_neuron_clustering_metric(
        model=model_orig,
        target_layer=layer_path,
        val_loader=dataloader,
        k=100,
        patch_size=64,
        experiment_name=experiment_name,
        save_dir="/scratch/inf0/user/dbagci/pure",
        device='cuda',
        sae=model,
        min_cluster_size=5,
        min_samples=3,
        recompute_patches=False,
        recompute_activations=False
    )
    
    # metrics = m.compute_neuron_clustering_metric_optimized(
    #     model=model_orig,
    #     target_layer=layer_path,
    #     val_loader=dataloader,
    #     k=100,
    #     patch_size=64,
    #     experiment_name=experiment_name,
    #     save_dir="/scratch/inf0/user/dbagci/pure",
    #     device='cuda',
    #     sae=model,
    #     min_cluster_size=5,
    #     min_samples=3,
    #     recompute_patches=False,
    #     recompute_activations=False,
    #     clip_batch_size=100,
    #     neurons_per_batch=5
    # )

    # # Run again - will load from cache instantly
    # metrics = m.neuron_activation_clustering_metric(
    #     model=model_orig,
    #     sae=model,
    #     target_layer=layer_path,
    #     val_loader=dataloader,
    #     experiment_name=experiment_name,  # Same name loads cache
    #     use_cache=True
    # )

    
    results = {
         "Model Name": [args.model],
         "Intra Score": [metrics[0]],
         "Inter Score": [metrics[1]],
         "Ratio": [metrics[2]]
    }
    
    df = pd.DataFrame(results)

    # # Append instead of overwrite
    csv_file = "/BS/disentanglement/work/Disentanglement/MSAE/results_ms_score.csv"
    df.to_csv(csv_file, mode="a", index=False, header=not os.path.exists(csv_file))
    
    # print(results)


if __name__ == "__main__":
    args = parse_args()
    main(args)