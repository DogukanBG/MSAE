import os
import torch
import logging
import argparse
import numpy as np
from tqdm import tqdm
import random
import wandb
import gc
from pathlib import Path
from sae import load_model
from utils import SAEDataset, set_seed, get_device
from metrics import (
    explained_variance_full,
    normalized_mean_absolute_error,
    l0_messure,
    cknna,
    orthogonal_decoder
)
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract and evaluate representations from SAE models")
    parser.add_argument("-m", "--model", type=str, required=True, help="Path to the trained model file (.pt)")
    parser.add_argument("-d", "--data", type=str, required=True, help="Path to the dataset file (.npy)")
    parser.add_argument("-b", "--batch-size", type=int, default=10000, help="Batch size for processing data")
    parser.add_argument("-o", "--output-path", type=str, default=".", help="Directory path to save extracted representations")
    parser.add_argument("-s", "--seed", type=int, default=42, help="Random seed for reproducibility")
    
    # W&B arguments
    parser.add_argument("--wandb-project", type=str, default="sae-evaluation", help="W&B project name")
    parser.add_argument("--wandb-entity", type=str, default=None, help="W&B entity/team name")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="W&B run name (auto-generated if not provided)")
    parser.add_argument("--wandb-tags", type=str, nargs="+", default=[], help="Tags for W&B run")
    parser.add_argument("--wandb-notes", type=str, default="", help="Notes for W&B run")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
    
    # Monosemanticity metric arguments
    parser.add_argument("--compute-monosemanticity", action="store_true", help="Compute monosemanticity clustering metrics")
    parser.add_argument("--imagenet-path", type=str, default="/scratch/inf0/user/mparcham/ILSVRC2012/val_categorized", 
                        help="Path to ImageNet validation set")
    parser.add_argument("--log-top-patches", action="store_true", help="Log top-k activating patches to W&B")
    parser.add_argument("--num-patches-to-log", type=int, default=5, help="Number of top patches to log per neuron")
    parser.add_argument("--num-neurons-to-log", type=int, default=20, help="Number of neurons to log patches for")
    
    return parser.parse_args()


def test_size(model: str) -> float:
    splits = model.split("_")
    ef = int(float(splits[0]) / float(splits[1]))
    return ef


def get_model(model_name: str, device: str):
    """Returns a model for image embeddings."""
    if model_name.lower() == "resnet50":
        model = torchvision.models.resnet50(pretrained=True)
        embedding_dim = 2048
        layer_path = "layer4.2.conv3"
    elif model_name.lower() == "convnext_tiny":
        model = torchvision.models.convnext_tiny(pretrained=True)
        embedding_dim = 768
        layer_path = None
    elif model_name.lower() == "vit_b_16":
        model = torchvision.models.vit_b_16(pretrained=True)
        embedding_dim = 768
        layer_path = "encoder.layers.encoder_layer_11.mlp.3"
    elif model_name.lower() == "dinov2_vitl14":
        import torch.hub
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
        embedding_dim = 1024
        layer_path = "blocks.23.mlp.fc2"
    elif model_name.lower() == "dinov2_vitb14":
        import torch.hub
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        embedding_dim = 768
        layer_path = "blocks.11.mlp.fc2"
    else:
        raise ValueError(f"Unsupported model {model_name}")
    
    model.to(device).eval()
    return model, layer_path, embedding_dim


def create_data_loaders(root: str, batch_size: int = 256, num_workers: int = 0, image_size: int = 224):
    """Create ImageNet data loaders."""
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = ImageFolder(root=root, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, 
                       num_workers=num_workers, pin_memory=True)
    return loader, len(dataset)


def log_top_activating_patches(top_k_patches_dict, num_neurons=20, num_patches=5, 
                               neuron_indices=None):
    """
    Log top activating patches for selected neurons to W&B.
    
    Args:
        top_k_patches_dict: Dictionary mapping neuron_idx -> list of PIL images
        num_neurons: Number of neurons to visualize
        num_patches: Number of patches to show per neuron
        neuron_indices: Specific neuron indices to log (if None, samples randomly)
    """
    if wandb.run is None:
        return
    
    total_neurons = len(top_k_patches_dict)
    
    # Select neurons to visualize
    if neuron_indices is None:
        # Sample uniformly across all neurons
        step = max(1, total_neurons // num_neurons)
        neuron_indices = list(range(0, total_neurons, step))[:num_neurons]
    
    logger.info(f"Logging top-{num_patches} patches for {len(neuron_indices)} neurons to W&B...")
    
    # Create a table for organized viewing
    columns = ["neuron_idx"] + [f"patch_{i+1}" for i in range(num_patches)]
    table = wandb.Table(columns=columns)
    
    for neuron_idx in neuron_indices:
        if neuron_idx not in top_k_patches_dict:
            continue
            
        patches = top_k_patches_dict[neuron_idx][:num_patches]
        
        # Convert patches to wandb.Image objects
        row = [neuron_idx]
        for patch in patches:
            row.append(wandb.Image(patch, caption=f"Neuron {neuron_idx}"))
        
        # Pad if we have fewer patches than expected
        while len(row) < len(columns):
            row.append(None)
            
        table.add_data(*row)
    
    wandb.log({"top_activating_patches": table})
    
    # Also create a panel view for better visualization
    images_dict = {}
    for neuron_idx in neuron_indices[:10]:  # Limit to 10 for panel view
        if neuron_idx not in top_k_patches_dict:
            continue
        patches = top_k_patches_dict[neuron_idx][:num_patches]
        images_dict[f"neuron_{neuron_idx}"] = [wandb.Image(p) for p in patches]
    
    if images_dict:
        wandb.log({"patch_grid": images_dict})

def extract_sae_type(model_path_name):
    if "ReLU_003" in model_path_name:
        return "ReLU_003"
    elif "TopKReLU_64_RW_" in model_path_name:
        return "TopKReLU_64_RW"
    elif "TopKReLU_64_UW_" in model_path_name:
        return "TopKReLU_64_UW"
    elif "TopKReLU_64_UW_" in model_path_name:
        return "TopKReLU_64_UW"
    elif "TopKReLU_256" in model_path_name and "batchtopk" in model_path_name:
        return "BatchTopKReLU_256"
    elif "TopKReLU_64" in model_path_name and "batchtopk" in model_path_name:
        return "BatchTopKReLU_64"
    elif "TopKReLU_256" in model_path_name and not "batchtopk" in model_path_name:
        return "TopKReLU_256"
    elif "TopKReLU_64" in model_path_name and not "batchtopk" in model_path_name:
        return "TopKReLU_64"

def get_representation(model_path_name, model, dataset, repr_file_name, batch_size, log_to_wandb=True):
    """
    Extract representations and evaluate model with W&B logging.
    """
    device = get_device()
    logger.info(f"Using device: {device}")
    
    model.eval()
    model.to(device)
    
    sae_type = extract_sae_type(model_path_name)
    
    # Log model architecture to W&B
    if log_to_wandb and wandb.run is not None:
        wandb.watch(model, log="all", log_freq=100)
    
    with torch.no_grad():
        # Prepare memory-mapped files
        # repr_file_name_output = f"{repr_file_name}_output_{len(dataset)}_{model.n_inputs}.npy"
        # memmap_output = np.memmap(repr_file_name_output, dtype='float32', mode='w+', 
        #                            shape=(len(dataset), model.n_inputs))
        
        # repr_file_name_repr = f"{repr_file_name}_repr_{len(dataset)}_{model.n_latents}.npy"
        # memmap_repr = np.memmap(repr_file_name_repr, dtype='float32', mode='w+', 
        #                         shape=(len(dataset), model.n_latents))
        
        # Create dataloader
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                                 shuffle=True, num_workers=0)
        
        # Metrics lists
        metrics_dict = {
            'l0': [], 'mae': [], 'fvu': [], 'cs': [], 'local_cknnas': [], 'global_cknnas': [],
            'sparse_l0': [], 'sparse_mae': [], 'sparse_fvu': [], 'sparse_cs': [], 'local_sparse_cknnas': [], 'global_sparse_cknnas': [],
            'local_hidden_cknnas': [], 'global_hidden_cknnas': [], 'local_hidden_sparse_cknnas': [], 'global_hidden_sparse_cknnas': []
        }
        dead_neurons_count = None
        
        # Process batches
        for idx, batch in enumerate(tqdm(dataloader, desc="Extracting representations")):
            start = batch_size * idx
            end = start + batch.shape[0]
            batch = batch.to(device)
            
            with torch.no_grad():
                sparse_outputs, sparse_representation, outputs, representations = model(batch)
            
            # Unprocess data
            # batch = dataset.unprocess_data(batch.cpu()).to(device)
            # outputs = dataset.unprocess_data(outputs.cpu()).to(device)
            # sparse_outputs = dataset.unprocess_data(sparse_outputs.cpu()).to(device)
            
            # Save to memmap
            # memmap_output[start:end] = outputs.cpu().numpy()
            # memmap_output.flush()
            # memmap_repr[start:end] = representations.cpu().numpy()
            # memmap_repr.flush()
            
            # Calculate metrics
            metrics_dict['fvu'].append(explained_variance_full(batch, outputs))
            metrics_dict['mae'].append(normalized_mean_absolute_error(batch, outputs))
            metrics_dict['cs'].append(torch.nn.functional.cosine_similarity(batch, outputs))
            metrics_dict['l0'].append(l0_messure(representations))
            
            if batch.shape[0] == batch_size:
                metrics_dict['local_cknnas'].append(cknna(batch, representations, topk=5))
                metrics_dict['global_cknnas'].append(cknna(batch, representations, topk=None))
                
                metrics_dict['local_hidden_cknnas'].append(cknna(batch, outputs, topk=5))
                metrics_dict['global_hidden_cknnas'].append(cknna(batch, outputs, topk=None))
            
            metrics_dict['sparse_fvu'].append(explained_variance_full(batch, sparse_outputs))
            metrics_dict['sparse_mae'].append(normalized_mean_absolute_error(batch, sparse_outputs))
            metrics_dict['sparse_cs'].append(torch.nn.functional.cosine_similarity(batch, sparse_outputs))
            metrics_dict['sparse_l0'].append(l0_messure(sparse_representation))
            
            if batch.shape[0] == batch_size:
                metrics_dict['local_sparse_cknnas'].append(cknna(batch, sparse_representation, topk=5))
                metrics_dict['global_sparse_cknnas'].append(cknna(batch, sparse_representation, topk=None))
                
                metrics_dict['local_hidden_sparse_cknnas'].append(cknna(batch, sparse_outputs, topk=5))
                metrics_dict['global_hidden_sparse_cknnas'].append(cknna(batch, sparse_outputs, topk=None))
            
            # Track dead neurons
            if dead_neurons_count is None:
                dead_neurons_count = (representations != 0).sum(dim=0).cpu().long()
            else:
                dead_neurons_count += (representations != 0).sum(dim=0).cpu().long()
            
            # Log batch metrics to W&B
            if log_to_wandb and wandb.run is not None and idx % 10 == 0:
                wandb.log({
                    "batch_idx": idx,
                    "batch/sparse_fvu": metrics_dict['sparse_fvu'][-1].mean().item(),
                    "batch/sparse_l0": metrics_dict['sparse_l0'][-1].mean().item(),
                })
        
        # Aggregate metrics
        aggregated = {}
        for key in ['mae', 'cs', 'l0', 'fvu', 'sparse_mae', 'sparse_cs', 'sparse_l0', 'sparse_fvu']:
            aggregated[key] = torch.cat(metrics_dict[key], dim=0).cpu().numpy()
        
        aggregated['local_cknnas'] = np.array(metrics_dict['local_cknnas'])
        aggregated['global_cknnas'] = np.array(metrics_dict['global_cknnas'])
        aggregated['local_sparse_cknnas'] = np.array(metrics_dict['local_sparse_cknnas'])
        aggregated['global_sparse_cknnas'] = np.array(metrics_dict['global_sparse_cknnas'])
        
        aggregated['local_hidden_cknnas'] = np.array(metrics_dict['local_hidden_cknnas'])
        aggregated['global_hidden_cknnas'] = np.array(metrics_dict['global_hidden_cknnas'])
        aggregated['local_hidden_sparse_cknnas'] = np.array(metrics_dict['local_hidden_sparse_cknnas'])
        aggregated['global_hidden_sparse_cknnas'] = np.array(metrics_dict['global_hidden_sparse_cknnas'])
        
        number_of_dead_neurons = torch.where(dead_neurons_count == 0)[0].shape[0]
        do = orthogonal_decoder(model.decoder)
        
        # Compute final metrics
        final_metrics = {
            "model_name": model_path_name,
            "SAE_TYPE": sae_type,
            "num_dead_neurons": int(number_of_dead_neurons),
            #"dead_neurons_pct": round(100 * number_of_dead_neurons / model.n_latents, 2),
            "fvu_mean": float(np.mean(aggregated['sparse_fvu'])),
            #"fvu_std": float(np.std(aggregated['sparse_fvu'])),
            "mae_mean": float(np.mean(aggregated['sparse_mae'])),
            #"mae_std": float(np.std(aggregated['sparse_mae'])),
            "cosine_sim_mean": float(np.mean(aggregated['sparse_cs'])),
            #"cosine_sim_std": float(np.std(aggregated['sparse_cs'])),
            "l0_mean": float(np.mean(aggregated['sparse_l0'])),
            #"l0_std": float(np.std(aggregated['sparse_l0'])),
            
            "local_cknna_mean": float(np.mean(aggregated['local_sparse_cknnas'])),
            #"local_cknna_std": float(np.std(aggregated['local_sparse_cknnas'])),
            "global_cknna_mean": float(np.mean(aggregated['global_sparse_cknnas'])),
            #"global_cknna_std": float(np.std(aggregated['global_sparse_cknnas'])),
            "local_hidden_cknna_mean": float(np.mean(aggregated['local_hidden_sparse_cknnas'])),
            #"local_cknna_std": float(np.std(aggregated['local_sparse_cknnas'])),
            "global_hidden_cknna_mean": float(np.mean(aggregated['global_hidden_sparse_cknnas'])),
            #"global_cknna_std": float(np.std(aggregated['global_sparse_cknnas'])),
            
            "decoder_orthogonality": float(do),
            "expansion_f": int(int(model.n_latents)/int(model.n_inputs)),
        }
        
        # Log to console
        logger.info("\n" + "="*50)
        logger.info("FINAL EVALUATION METRICS")
        logger.info("="*50)
        for key, value in final_metrics.items():
            logger.info(f"{key}: {value}")
        logger.info("="*50 + "\n")
        
        # Log to W&B
        if log_to_wandb and wandb.run is not None:
            wandb.log(final_metrics)
            
            # Create summary metrics
            wandb.run.summary.update(final_metrics)
            
            # Log histograms
            # wandb.log({
            #     "histograms/sparse_fvu": wandb.Histogram(aggregated['sparse_fvu']),
            #     "histograms/sparse_l0": wandb.Histogram(aggregated['sparse_l0']),
            #     "histograms/dead_neurons": wandb.Histogram(dead_neurons_count.cpu().numpy()),
            # })
            
            # Save artifacts
            #artifact = wandb.Artifact(f"representations-{model_path_name}", type="representations")
            #artifact.add_file(repr_file_name_output)
            #artifact.add_file(repr_file_name_repr)
            #wandb.log_artifact(artifact)
        
        return final_metrics


def main(args):
    set_seed(args.seed)
    
    # Extract model and data names
    model_path_name = args.model.split("/")[-1].replace(".pt", "")
    if "batchtopk" in args.model:
        model_path_name += "_batchtopk"
    
    data_path_name = args.data.split("/")[-1].replace(".npy", "")
    
    expansion_factor = test_size(model_path_name)
    
    # Determine model type
    model_type = None
    if "resnet50" in args.data:
        model_type = "resnet50"
    elif "vit_b_16" in args.data:
        model_type = "vit_b_16"
    elif "dinov2_vitl14" in args.data:
        model_type = "dinov2_vitl14"
    elif "dinov2_vitb14" in args.data:
        model_type = "dinov2_vitb14"
    
    # Initialize W&B
    use_wandb = not args.no_wandb
    if use_wandb:
        run_name = args.wandb_run_name or f"{model_path_name}_{data_path_name}"
        
        config = {
            "model_path": args.model,
            "data_path": args.data,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "model_type": model_type,
            "data_name": data_path_name,
            "expansion_factor": expansion_factor,
            "compute_monosemanticity": args.compute_monosemanticity,
        }
        
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            group=model_type,
            name=run_name,
            tags=args.wandb_tags + ([model_type] if model_type else []),
            notes=args.wandb_notes,
            config=config
        )
        logger.info(f"W&B run initialized: {wandb.run.name}")
    
    try:
        # Load model
        model, mean_center, scaling_factor, target_norm = load_model(args.model)
        logger.info("Model loaded")
        
        # Load dataset
        if ("text" in args.model and "text" in args.data) or ("image" in args.model and "image" in args.data):
            dataset = SAEDataset(args.data, split="val", model_type=model_type)
            dataset.mean = mean_center.cpu()
            dataset.scaling_factor = scaling_factor
        else:
            dataset = SAEDataset(
                args.data,
                mean_center=mean_center.sum() != 0.0,
                target_norm=target_norm,
                split="val",
                model_type=model_type
            )
        
        logger.info(f"Dataset loaded with length: {len(dataset)}")
        
        # Log dataset info to W&B
        if use_wandb:
            wandb.config.update({
                "dataset_size": len(dataset),
                "mean_center": float(dataset.mean.mean()),
                "scaling_factor": float(dataset.scaling_factor),
                "target_norm": float(dataset.target_norm) if dataset.target_norm else None,
                "n_latents": int(model.n_latents),
                "n_inputs": int(model.n_inputs),
            })
        
        # Extract representations and evaluate
        repr_file_name = os.path.join(args.output_path, f"{data_path_name}_{model_path_name}")
        final_metrics = {}
        if args.compute_monosemanticity:
            final_metrics = get_representation(
                model_path_name, model, dataset, repr_file_name, 
                args.batch_size, log_to_wandb=use_wandb
            )
        
        # Compute monosemanticity metrics if requested
        if args.compute_monosemanticity and model_type is not None:
            logger.info("\n" + "="*50)
            logger.info("Computing monosemanticity metrics...")
            logger.info("="*50)
            
            # Load original model for activation extraction
            model_orig, layer_path, embedding_dim = get_model(
                model_name=model_type, 
                device=get_device()
            )
            
            # Create ImageNet dataloader
            dataloader, _ = create_data_loaders(
                args.imagenet_path, 
                num_workers=0
            )
            
            # Import metrics module
            import metrics as m
            
            # Compute clustering metrics
            avg_intra, avg_inter, sep_ratio = m.compute_neuron_clustering_metric(
                model=model_orig,
                target_layer=layer_path,
                val_loader=dataloader,
                k=100,
                patch_size=64,
                experiment_name=model_path_name,
                device=get_device(),
                n_workers=4,
                recompute_embeddings=False,
                sae=model,  
                return_top_patches=args.log_top_patches  
            )
            
            logger.info(f"\nMonosemanticity Metrics:")
            logger.info(f"Average Intra-cluster Distance: {avg_intra:.6f}")
            logger.info(f"Average Inter-cluster Distance: {avg_inter:.6f}")
            logger.info(f"Separation Ratio: {sep_ratio:.6f}")
            
            # Log to W&B
            if use_wandb:
                mono_metrics = {
                    "expansion_f": int(int(model.n_latents)/int(model.n_inputs)),
                    "avg_intra": float(avg_intra),
                    "avg_inter": float(avg_inter),
                    "sep_ratio": float(sep_ratio),
                }
                wandb.log(mono_metrics)
                wandb.run.summary.update(mono_metrics)
                
                # Log top activating patches if requested
                # if args.log_top_patches and top_k_patches is not None:
                #     log_top_activating_patches(
                #         top_k_patches,
                #         num_neurons=args.num_neurons_to_log,
                #         num_patches=args.num_patches_to_log
                #     )
            
            # Add to final metrics for combined logging
            final_metrics.update({
                "avg_intra_cluster": float(avg_intra),
                "avg_inter_cluster": float(avg_inter),
                "separation_ratio": float(sep_ratio),
            })
            
            # Cleanup
            del model_orig, dataloader
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        logger.info("\n" + "="*50)
        logger.info("Evaluation completed successfully!")
        logger.info("="*50)
        
        # Create summary table
        if use_wandb:
            summary_data = []
            for key, value in final_metrics.items():
                if isinstance(value, (int, float)):
                    summary_data.append([key, value])
            
            summary_table = wandb.Table(columns=["Metric", "Value"], data=summary_data)
            wandb.log({"final_metrics_summary": summary_table})
        
    except Exception as e:
        logger.error(f"Error during evaluation: {e}", exc_info=True)
        if use_wandb:
            wandb.finish(exit_code=1)
        raise
    
    finally:
        if use_wandb:
            wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    main(args)