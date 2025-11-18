import torch
import inspect
import argparse
import logging
from tqdm import tqdm
from dataclasses import asdict
import wandb
import os

from metrics import calculate_similarity_metrics, identify_dead_neurons, orthogonal_decoder, cknna, explained_variance
from utils import SAEDataset, set_seed, get_device, geometric_median, calculate_vector_mean, LinearDecayLR, CosineWarmupScheduler
from config import get_config
from sae import Autoencoder, MatryoshkaAutoencoder, Transcoder
from loss import SAELoss,TCLoss
import utils as ut 

"""
Sparse Autoencoder (SAE) Training Script with WandB Integration

This script provides a complete pipeline for training various types of sparse autoencoder models,
including standard SAEs with different activation functions and Matryoshka SAEs with nested
feature hierarchies. It handles training, evaluation, and model saving with configurable
hyperparameters, and comprehensive Weights & Biases logging.
"""

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments for the SAE training script.
    
    Returns:
        argparse.Namespace: Parsed command line arguments with the following fields:
            - dataset_train: Path to the training dataset
            - dataset_test: Path to the testing/validation dataset
            - model: Model architecture to train (e.g., "ReLUSAE", "TopKSAE")
            - activation: Activation function to use
            - epochs: Number of training epochs
            - learning_rate: Initial learning rate
            - expansion_factor: Ratio of latent dimensions to input dimensions
    """
    parser = argparse.ArgumentParser(description="Train Sparse Autoencoder (SAE) models")
    parser.add_argument("-dt", "--dataset_train", type=str, required=True, 
                       help="Path to training dataset file (.npy)")
    parser.add_argument("-ds", "--dataset_test", type=str, required=True, 
                       help="Path to testing/validation dataset file (.npy)")
    parser.add_argument("-dm", "--dataset_second_modality", type=str, default=None,
                       help="Path to second modality dataset file (.npy)")
    parser.add_argument("-m", "--model", type=str, required=True, 
                       choices=["ReLUSAE", "ReLUTC", "TopKSAE", "BatchTopKSAE", "MSAE_UW", "MSAE_RW"], 
                       help="Model architecture to train")
    parser.add_argument("-a", "--activation", type=str, required=True, 
                       help="Activation function (e.g., 'ReLU_003', 'TopKReLU_64')")
    parser.add_argument("-e", "--epochs", type=int, default=100, 
                       help="Number of training epochs")
    parser.add_argument("-ef", "--expansion_factor", type=float, default=1.0, 
                       help="Ratio of latent dimensions to input dimensions")
    parser.add_argument("--lr", type=float, default=0.00005, 
                       help="Ratio of latent dimensions to input dimensions")
    parser.add_argument("--train_tc", action='store_true',
                       help="Train transcoder")
    parser.add_argument("--eval_only", default=False, type=bool)
    parser.add_argument("-bs", "--batch_size", type=int, default=8192, 
                       help="Number of training epochs")
    parser.add_argument("--wandb_project", type=str, default="Sparse Autoencoders",
                       help="WandB project name")
    parser.add_argument("--wandb_entity", type=str, default=None,
                       help="WandB entity (username or team)")
    parser.add_argument("--wandb_mode", type=str, default="online",
                       choices=["online", "offline", "disabled"],
                       help="WandB logging mode")
    parser.add_argument("--wandb_tags", type=str, nargs="+", default=[],
                       help="Additional tags for WandB run")
    
    
    parser.add_argument("--eval_hc", default=False, type=bool)
    parser.add_argument("--tau", type=float, default=0.01)
    parser.add_argument("--model_orig", type=str)
    
    
    return parser.parse_args()

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

def extract_model_type(dataset_path: str) -> str:
    """Extract model type from dataset path."""
    if "resnet50" in dataset_path.lower():
        return "ResNet50"
    elif "vit_b_16" in dataset_path.lower():
        return "ViT-B/16"
    elif "dinov2_vitb14" in dataset_path.lower():
        return "DINOv2-ViT-B/14"
    elif "dinov2_vitl14" in dataset_path.lower():
        return "DINOv2-ViT-L/14"
    elif "convnext" in dataset_path.lower():
        return "ConvNeXt"
    else:
        return "Unknown"

def create_wandb_run_name(cfg, args, model_type: str) -> str:
    """Create a descriptive and unique run name for WandB."""
    activation = args.activation
    if args.model == "ReLUSAE" and "_" in args.activation:
        activation_base, sparse_weight = args.activation.split("_")
        activation = f"{activation_base}_L1-0.{sparse_weight}"
    
    if cfg.model.use_matryoshka:
        matryoshka_type = "RW" if "RW" in args.model else "UW"
        run_name = f"{model_type}_{args.model}_{matryoshka_type}_{activation}_EF{args.expansion_factor}"
    else:
        run_name = f"{model_type}_{args.model}_{activation}_EF{args.expansion_factor}"
    
    return run_name


def eval(model, eval_loader, loss_fn, device, cfg, prefix="val"):
    """Evaluation function that returns metrics dict for WandB logging."""
    # Evaluation phase
    loss_all = 0.0
    recon_loss_all = 0.0
    sparse_loss_all = 0.0
    cknna_score_sparse_sum = 0.0
    cknna_score_all_sum = 0.0
    fvu_score_all_sum = 0.0
    fvu_score_sparse_sum = 0.0
    diagonal_cs_sparse_sum = 0.0
    diagonal_cs_all_sum = 0.0
    mae_distance_sparse_sum = 0.0
    mae_distance_all_sum = 0.0
    od_sum = 0.0
    sparsity_sparse_sum = 0.0
    sparsity_all_sum = 0.0
    
    # Switch to evaluation mode
    model.eval()
    for step, embeddings in enumerate(tqdm(eval_loader, desc=f"Evaluation ({prefix})")):
        embeddings = embeddings.to(device)
        
        # Forward pass without gradient computation
        with torch.no_grad():
            recons_sparse, repr_sparse, recons_all, repr_all = model(embeddings)
            loss, recon_loss, sparse_loss = loss_fn(recons_all, embeddings, repr_all)
        
        if cfg.model.use_matryoshka:
            recons_sparse = recons_sparse[0]
            repr_sparse = repr_sparse[0]
        
        # Accumulate loss metrics
        loss_all += loss.item()
        recon_loss_all += recon_loss.item()
        sparse_loss_all += sparse_loss.item()
        
        # Accumulate CKNNA scores
        cknna_score_sparse_sum += cknna(recons_sparse, embeddings)
        cknna_score_all_sum += cknna(recons_all, embeddings)
        
        # Accumulate similarity metrics
        fvu_score_sparse_sum += explained_variance(embeddings, recons_sparse)
        fvu_score_all_sum += explained_variance(embeddings, recons_all)
        distance_sparse = calculate_similarity_metrics(embeddings, recons_sparse)
        distance_all = calculate_similarity_metrics(embeddings, recons_all)
        diagonal_cs_sparse_sum += distance_sparse[0]
        mae_distance_sparse_sum += distance_sparse[1]
        diagonal_cs_all_sum += distance_all[0]
        mae_distance_all_sum += distance_all[1]
        
        # Accumulate orthogonality measure
        od_sum += orthogonal_decoder(model.decoder)
        
        # Accumulate sparsity measures
        sparsity_sparse_sum += (repr_sparse == 0.0).float().mean(axis=-1).mean()
        sparsity_all_sum += (repr_all == 0.0).float().mean(axis=-1).mean()
    
    # Calculate averages
    num_batches = len(eval_loader)
    metrics = {
        f"{prefix}/loss": loss_all / num_batches,
        f"{prefix}/reconstruction_loss": recon_loss_all / num_batches,
        f"{prefix}/sparsity_loss": sparse_loss_all / num_batches,
        f"{prefix}/fvu_sparse": fvu_score_sparse_sum / num_batches,
        f"{prefix}/fvu_all": fvu_score_all_sum / num_batches,
        f"{prefix}/cknna_sparse": cknna_score_sparse_sum / num_batches,
        f"{prefix}/cknna_all": cknna_score_all_sum / num_batches,
        f"{prefix}/cosine_sim_sparse": diagonal_cs_sparse_sum / num_batches,
        f"{prefix}/mae_sparse": mae_distance_sparse_sum / num_batches,
        f"{prefix}/cosine_sim_all": diagonal_cs_all_sum / num_batches,
        f"{prefix}/mae_all": mae_distance_all_sum / num_batches,
        f"{prefix}/sparsity_sparse": sparsity_sparse_sum / num_batches,
        f"{prefix}/sparsity_all": sparsity_all_sum / num_batches,
        f"{prefix}/orthogonal_decoder": od_sum / num_batches,
    }
    
    # Log evaluation metrics
    logger.info(f"Evaluation results ({prefix}):")
    logger.info(f"  Loss: {metrics[f'{prefix}/loss']:.6f}")
    logger.info(f"  Reconstruction Loss: {metrics[f'{prefix}/reconstruction_loss']:.6f}")
    logger.info(f"  Sparsity Loss: {metrics[f'{prefix}/sparsity_loss']:.6f}")
    logger.info(f"  FVU Sparse: {metrics[f'{prefix}/fvu_sparse']:.4f}")
    logger.info(f"  FVU All: {metrics[f'{prefix}/fvu_all']:.4f}")
    logger.info(f"  CKNNA Sparse: {metrics[f'{prefix}/cknna_sparse']:.4f}")
    logger.info(f"  CKNNA All: {metrics[f'{prefix}/cknna_all']:.4f}")
    logger.info(f"  Cosine Similarity Sparse: {metrics[f'{prefix}/cosine_sim_sparse']:.4f}")
    logger.info(f"  MAE Distance Sparse: {metrics[f'{prefix}/mae_sparse']:.4f}")
    logger.info(f"  Cosine Similarity All: {metrics[f'{prefix}/cosine_sim_all']:.4f}")
    logger.info(f"  MAE Distance All: {metrics[f'{prefix}/mae_all']:.4f}")
    logger.info(f"  Sparsity Sparse: {metrics[f'{prefix}/sparsity_sparse']:.4f}")
    logger.info(f"  Sparsity All: {metrics[f'{prefix}/sparsity_all']:.4f}")
    logger.info(f"  Orthogonal Decoder Loss: {metrics[f'{prefix}/orthogonal_decoder']:.6f}")
    
    return metrics

def eval_tc(model, eval_loader_input, eval_loader_output, loss_fn, device, cfg, prefix="val"):
    """Evaluation function that returns metrics dict for WandB logging."""
    # Evaluation phase
    loss_all = 0.0
    recon_loss_all = 0.0
    sparse_loss_all = 0.0
    cknna_score_sparse_sum = 0.0
    cknna_score_all_sum = 0.0
    fvu_score_all_sum = 0.0
    fvu_score_sparse_sum = 0.0
    diagonal_cs_sparse_sum = 0.0
    diagonal_cs_all_sum = 0.0
    mae_distance_sparse_sum = 0.0
    mae_distance_all_sum = 0.0
    od_sum = 0.0
    sparsity_sparse_sum = 0.0
    sparsity_all_sum = 0.0
    
    # Switch to evaluation mode
    model.eval()
    for step, (input_embeddings, output_embeddings) in enumerate(tqdm(zip(eval_loader_input, eval_loader_output), desc=f"Evaluation ({prefix})")):
        input_embeddings = input_embeddings.to(device)
        output_embeddings = output_embeddings.to(device)
        
        # Forward pass without gradient computation
        with torch.no_grad():
            recons_sparse, repr_sparse, recons_all, repr_all = model(input_embeddings)
            loss, recon_loss, sparse_loss = loss_fn(recons_all, output_embeddings, repr_all)
        
        if cfg.model.use_matryoshka:
            recons_sparse = recons_sparse[0]
            repr_sparse = repr_sparse[0]
        
        # Accumulate loss metrics
        loss_all += loss.item()
        recon_loss_all += recon_loss.item()
        sparse_loss_all += sparse_loss.item()
        
        # Accumulate CKNNA scores
        cknna_score_sparse_sum += cknna(recons_sparse, output_embeddings)
        cknna_score_all_sum += cknna(recons_all, output_embeddings)
        
        # Accumulate similarity metrics
        fvu_score_sparse_sum += explained_variance(output_embeddings, recons_sparse)
        fvu_score_all_sum += explained_variance(output_embeddings, recons_all)
        distance_sparse = calculate_similarity_metrics(output_embeddings, recons_sparse)
        distance_all = calculate_similarity_metrics(output_embeddings, recons_all)
        diagonal_cs_sparse_sum += distance_sparse[0]
        mae_distance_sparse_sum += distance_sparse[1]
        diagonal_cs_all_sum += distance_all[0]
        mae_distance_all_sum += distance_all[1]
        
        # Accumulate orthogonality measure
        od_sum += orthogonal_decoder(model.decoder)
        
        # Accumulate sparsity measures
        sparsity_sparse_sum += (repr_sparse == 0.0).float().mean(axis=-1).mean()
        sparsity_all_sum += (repr_all == 0.0).float().mean(axis=-1).mean()
    
    # Calculate averages
    num_batches = len(eval_loader_input)
    metrics = {
        f"{prefix}/loss": loss_all / num_batches,
        f"{prefix}/reconstruction_loss": recon_loss_all / num_batches,
        f"{prefix}/sparsity_loss": sparse_loss_all / num_batches,
        f"{prefix}/fvu_sparse": fvu_score_sparse_sum / num_batches,
        f"{prefix}/fvu_all": fvu_score_all_sum / num_batches,
        f"{prefix}/cknna_sparse": cknna_score_sparse_sum / num_batches,
        f"{prefix}/cknna_all": cknna_score_all_sum / num_batches,
        f"{prefix}/cosine_sim_sparse": diagonal_cs_sparse_sum / num_batches,
        f"{prefix}/mae_sparse": mae_distance_sparse_sum / num_batches,
        f"{prefix}/cosine_sim_all": diagonal_cs_all_sum / num_batches,
        f"{prefix}/mae_all": mae_distance_all_sum / num_batches,
        f"{prefix}/sparsity_sparse": sparsity_sparse_sum / num_batches,
        f"{prefix}/sparsity_all": sparsity_all_sum / num_batches,
        f"{prefix}/orthogonal_decoder": od_sum / num_batches,
    }
    
    # Log evaluation metrics
    logger.info(f"Evaluation results ({prefix}):")
    logger.info(f"  Loss: {metrics[f'{prefix}/loss']:.6f}")
    logger.info(f"  Reconstruction Loss: {metrics[f'{prefix}/reconstruction_loss']:.6f}")
    logger.info(f"  Sparsity Loss: {metrics[f'{prefix}/sparsity_loss']:.6f}")
    logger.info(f"  FVU Sparse: {metrics[f'{prefix}/fvu_sparse']:.4f}")
    logger.info(f"  FVU All: {metrics[f'{prefix}/fvu_all']:.4f}")
    logger.info(f"  CKNNA Sparse: {metrics[f'{prefix}/cknna_sparse']:.4f}")
    logger.info(f"  CKNNA All: {metrics[f'{prefix}/cknna_all']:.4f}")
    logger.info(f"  Cosine Similarity Sparse: {metrics[f'{prefix}/cosine_sim_sparse']:.4f}")
    logger.info(f"  MAE Distance Sparse: {metrics[f'{prefix}/mae_sparse']:.4f}")
    logger.info(f"  Cosine Similarity All: {metrics[f'{prefix}/cosine_sim_all']:.4f}")
    logger.info(f"  MAE Distance All: {metrics[f'{prefix}/mae_all']:.4f}")
    logger.info(f"  Sparsity Sparse: {metrics[f'{prefix}/sparsity_sparse']:.4f}")
    logger.info(f"  Sparsity All: {metrics[f'{prefix}/sparsity_all']:.4f}")
    logger.info(f"  Orthogonal Decoder Loss: {metrics[f'{prefix}/orthogonal_decoder']:.6f}")
    
    return metrics

def main(args):
    """
    Main training function for Sparse Autoencoders with WandB integration.
    
    This function handles the complete training pipeline:
    1. Setting up configuration based on model type and arguments
    2. Loading and preparing datasets
    3. Initializing the appropriate model (standard or Matryoshka SAE)
    4. Setting up loss function, optimizer, and learning rate scheduler
    5. Executing the training loop with periodic evaluation
    6. Tracking metrics including reconstruction quality, sparsity, and dead neurons
    7. Saving the trained model with relevant metadata
    8. Logging all metrics and artifacts to Weights & Biases
    
    Args:
        args (argparse.Namespace): Command line arguments from parse_args()
    """
    logger.info("Starting training with the following arguments:")
    logger.info(args)
    
    # Get configuration based on model type
    cfg = get_config(args.model)
    cfg.training.epochs = args.epochs
    cfg.training.batch_size = args.batch_size
    cfg.training.num_workers = 4
    ##cfg.training.mean_center=True#cfg.training.mean_center, 
    ##cfg.training.target_norm=None#cfg.training.target_norm,
    
    # Set the random seed for reproducibility
    set_seed(79677) #V1: 68779 V2: 79677
    
    # Set the device (GPU/CPU)
    device = get_device()
    logger.info(f"Using device: {device}")
    
    # Extract model type from dataset path
    model_type_str = None
    if "resnet50" in args.dataset_train:
        model_type_str = "resnet50"
    elif "vit_b_16" in args.dataset_train:
        model_type_str = "vit_b_16"
    elif "dinov2_vitb14" in args.dataset_train:
        model_type_str = "dinov2_vitb14"
    elif "dinov2_vitl14" in args.dataset_train:
        model_type_str = "dinov2_vitl14"
    
    model_type_display = extract_model_type(args.dataset_train)
    
    # Load datasets
    train_ds = SAEDataset(
        args.dataset_train, 
        dtype=cfg.training.dtype, 
        mean_center=cfg.training.mean_center, 
        target_norm=cfg.training.target_norm,
        split="train",
        model_type=model_type_str
    )
    eval_ds = SAEDataset(
        args.dataset_test, 
        dtype=cfg.training.dtype, 
        mean_center=cfg.training.mean_center,
        target_norm=cfg.training.target_norm,
        split="val",
        model_type=model_type_str
    )
    logger.info(f"Training dataset length: {len(train_ds)}, Evaluation dataset length: {len(eval_ds)}, Embedding size: {train_ds.vector_size}")
    logger.info(f"Training dataset mean center: {train_ds.mean.mean()}, Scaling factor: {train_ds.scaling_factor} with target norm {train_ds.target_norm}")
    logger.info(f"Evaluation dataset mean center: {eval_ds.mean.mean()}, Scaling factor: {eval_ds.scaling_factor} with target norm {eval_ds.target_norm}")
    assert train_ds.vector_size == eval_ds.vector_size, "Training and evaluation datasets must have the same embedding size"
    
    has_second_modality = False
    if args.dataset_second_modality is not None:
        has_second_modality = True
        eval_ds_second = SAEDataset(
            args.dataset_second_modality,
            dtype=cfg.training.dtype,
            mean_center=cfg.training.mean_center,
            target_norm=cfg.training.target_norm
        )
        logger.info(f"Second modality dataset mean center: {eval_ds_second.mean.mean()}, Scaling factor: {eval_ds_second.scaling_factor} with target norm {eval_ds_second.target_norm}")
        assert train_ds.vector_size == eval_ds_second.vector_size, "Training and second modality datasets must have the same embedding size"
    
    # Set model parameters based on dataset and arguments
    cfg.model.n_inputs = train_ds.vector_size
    
    # Calculate number of latent dimensions using expansion factor
    cfg.model.n_latents = int(args.expansion_factor * train_ds.vector_size)
    logger.info(f"Expansion factor: {args.expansion_factor}, Latent dimensions: {cfg.model.n_latents}")
    
    # Extract l1 from ReLU if applied
    if args.model == "ReLUSAE" and "_" in args.activation:
        args.activation, sparse_weight = args.activation.split("_")
        cfg.loss.sparse_weight = float(f"0.{sparse_weight}")
        logger.info(f"Changing sparsity weight value to {cfg.loss.sparse_weight}")
        
    # Override activation if specified in arguments
    if args.activation:
        cfg.model.activation = args.activation
    
    # Configure Matryoshka SAE parameters if applicable
    if cfg.model.use_matryoshka:
        # Max nesting list
        if cfg.model.nesting_list > cfg.model.max_nesting:
            max_nesting = cfg.model.n_latents
        else:
            max_nesting = cfg.model.max_nesting
        
        # Generate nesting list if a single value was provided
        if isinstance(cfg.model.nesting_list, int):
            logger.info(f"Generating nesting list from {cfg.model.nesting_list} to {max_nesting}")
            start = [cfg.model.nesting_list]
            while start[-1] < max_nesting:
                new_k = start[-1] * 2
                if new_k > max_nesting:
                    break
                start.append(new_k)
            
            if max_nesting not in start:
                start.append(max_nesting)
            cfg.model.nesting_list = start
        
        # Set importance weights for different nesting levels
        if cfg.model.relative_importance == "RW":
            # Reverse weighting - higher weight for larger k values
            cfg.model.relative_importance = list(reversed(list(range(1, len(cfg.model.nesting_list)+1))))
        elif cfg.model.relative_importance == "UW":
            # Uniform weighting - equal weight for all k values
            cfg.model.relative_importance = [1.0] * len(cfg.model.nesting_list)

        logger.info(f"Using Matryoshka with nesting list: {cfg.model.nesting_list} and weighting function: {cfg.model.relative_importance}")
    else:
        logger.info(f"Using standard SAE with {cfg.model.activation} activation")
    
    # Initialize Weights & Biases
    wandb_run_name = create_wandb_run_name(cfg, args, model_type_display)
    
    # Prepare WandB config
    wandb_config = {
        # Model architecture
        "model_type": args.model,
        "backbone": model_type_display,
        "activation": cfg.model.activation,
        "expansion_factor": args.expansion_factor,
        "n_inputs": cfg.model.n_inputs,
        "n_latents": cfg.model.n_latents,
        "use_matryoshka": cfg.model.use_matryoshka,
        "tied_weights": cfg.model.tied,
        "normalize_decoder": cfg.model.normalize,
        "latent_soft_cap": cfg.model.latent_soft_cap,
        
        # Training hyperparameters
        "epochs": cfg.training.epochs,
        "batch_size": cfg.training.batch_size,
        "learning_rate": cfg.training.lr,
        "weight_decay": cfg.training.weight_decay,
        "beta1": cfg.training.beta1,
        "beta2": cfg.training.beta2,
        "eps": cfg.training.eps,
        "scheduler": cfg.training.scheduler,
        "clip_grad": cfg.training.clip_grad,
        
        # Loss configuration
        "reconstruction_loss": cfg.loss.reconstruction_loss,
        "sparse_loss": cfg.loss.sparse_loss,
        "sparse_weight": cfg.loss.sparse_weight,
        
        # Data preprocessing
        "mean_center": cfg.training.mean_center,
        "target_norm": cfg.training.target_norm,
        "bias_init_median": cfg.training.bias_init_median,
        
        # Dataset info
        "train_dataset": os.path.basename(args.dataset_train),
        "eval_dataset": os.path.basename(args.dataset_test),
        "train_size": len(train_ds),
        "eval_size": len(eval_ds),
        "has_second_modality": has_second_modality,
        
        # Device
        "device": str(device),
        "seed": 68779,
    }
    
    # Add Matryoshka-specific config
    if cfg.model.use_matryoshka:
        wandb_config.update({
            "nesting_list": cfg.model.nesting_list,
            "relative_importance": cfg.model.relative_importance,
            "matryoshka_type": "RW" if "RW" in args.model else "UW",
        })
    
    # Prepare tags
    tags = [
        args.model,
        model_type_display,
        cfg.model.activation,
        f"EF{args.expansion_factor}",
    ]
    if cfg.model.use_matryoshka:
        tags.append("Matryoshka")
        tags.append("RW" if "RW" in args.model else "UW")
    tags.extend(args.wandb_tags)
    
    # Initialize WandB run
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=wandb_run_name,
        config=wandb_config,
        tags=tags,
        group=model_type_display,  # Group by backbone model
        mode=args.wandb_mode,
    )
    
    logger.info(f"WandB run initialized: {wandb.run.name} (ID: {wandb.run.id})")
    logger.info(f"WandB URL: {wandb.run.url}")
    
    # Calculate bias initialization (median or zero)
    logger.info(f"Calculating bias initialization with median: {cfg.training.bias_init_median}")
    bias_init = 0.0
    if cfg.training.bias_init_median:
        # Use geometric median of a subset of data points for robustness
        bias_init = geometric_median(train_ds, device=device, max_number=len(train_ds)//10)
    logger.info(f"Bias initialization: {bias_init}")
    
    # Log bias initialization to WandB
    wandb.log({"initialization/bias_init_norm": torch.norm(bias_init).item() if isinstance(bias_init, torch.Tensor) else abs(bias_init)})
    
    # Initialize the appropriate model type
    if cfg.model.use_matryoshka:
        model = MatryoshkaAutoencoder(bias_init=bias_init, **asdict(cfg.model))
    else:
        model = Autoencoder(bias_init=bias_init, **asdict(cfg.model))
    model = model.to(device)
    
    # Log model architecture
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    wandb.log({
        "model/total_params": total_params,
        "model/trainable_params": trainable_params,
        "model/total_params_millions": total_params / 1e6,
    })
    
    # Prepare loss function
    # Use zeros or calculate mean from dataset depending on config
    mean_input = torch.zeros((train_ds.vector_size,), dtype=cfg.training.dtype)
    if not cfg.training.mean_center:
        mean_input = calculate_vector_mean(train_ds, num_workers=cfg.training.num_workers)
    
    mean_input = mean_input.to(device)
    loss_fn = SAELoss(
        reconstruction_loss=cfg.loss.reconstruction_loss,
        sparse_loss=cfg.loss.sparse_loss,
        sparse_weight=cfg.loss.sparse_weight,
        mean_input=mean_input,
    )
    
    # Prepare the optimizer with adaptive settings based on device
    fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and "cuda" in device.type
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=cfg.training.lr, 
        betas=(cfg.training.beta1, cfg.training.beta2), 
        eps=cfg.training.eps, 
        weight_decay=cfg.training.weight_decay, 
        fused=use_fused
    )
    
    logger.info(f"Using fused AdamW: {use_fused}")
    
    # Prepare the learning rate scheduler
    if cfg.training.scheduler == 1:
        # Linear decay scheduler
        scheduler = LinearDecayLR(optimizer, cfg.training.epochs, decay_time=cfg.training.decay_time)
    elif cfg.training.scheduler == 2:
        # Cosine annealing with warmup
        scheduler = CosineWarmupScheduler(
            optimizer, 
            max_lr=cfg.training.lr, 
            warmup_epoch=1, 
            final_lr_factor=0.1, 
            total_epochs=cfg.training.epochs
        )
    else:
        # No scheduler
        scheduler = None
    
    # Prepare the dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_ds, 
        batch_size=cfg.training.batch_size, 
        num_workers=cfg.training.num_workers, 
        shuffle=True
    )
    eval_loader = torch.utils.data.DataLoader(
        eval_ds, 
        batch_size=cfg.training.batch_size, 
        num_workers=cfg.training.num_workers, 
        shuffle=False
    )
    if has_second_modality:
        eval_loader_second = torch.utils.data.DataLoader(
            eval_ds_second,
            batch_size=cfg.training.batch_size,
            num_workers=cfg.training.num_workers,
            shuffle=False
        )

    # Training loop
    global_step = 0
    numb_of_dead_neurons = 0
    dead_neurons = []
    best_val_fvu = float('inf')  # Track best validation FVU for model saving
    
    for epoch in range(cfg.training.epochs):
        model.train()
        logger.info(f"Epoch {epoch+1}/{cfg.training.epochs}")
        
        # Epoch metrics accumulators
        epoch_loss_sum = 0.0
        epoch_recon_loss_sum = 0.0
        epoch_sparse_loss_sum = 0.0
        epoch_steps = 0
        
        # Training loop for current epoch
        for step, embeddings in enumerate(tqdm(train_loader, desc="Training")):
            optimizer.zero_grad()
            global_step += 1
            epoch_steps += 1
            embeddings = embeddings.to(device)
            
            # Forward pass through model
            recons_sparse, repr_sparse, recons_all, repr_all = model(embeddings)
            
            # Compute loss based on model type
            if cfg.model.use_matryoshka:
                # For Matryoshka models, compute weighted loss across all nesting levels
                sparse_loss = loss_fn(recons_all, embeddings, repr_all)[-1]
                recon_loss = loss_fn(recons_sparse[0], embeddings, repr_all)[1]
                
                # Weight reconstruction losses by relative importance
                loss_recon_all = torch.tensor(0., requires_grad=True, device=device)
                for i in range(len(recons_sparse)):
                    current_loss = loss_fn(recons_sparse[i], embeddings, repr_sparse[i])[1]
                    loss_recon_all = loss_recon_all + current_loss * model.relative_importance[i]

                # Normalize by sum of weights
                loss = loss_recon_all / sum(model.relative_importance)
                
                # Use first nesting level for metrics
                repr_sparse = repr_sparse[0]
                recons_sparse = recons_sparse[0]
            else:
                # Standard SAE loss computation
                loss, recon_loss, sparse_loss = loss_fn(recons_sparse, embeddings, repr_sparse)
            
            # Accumulate epoch metrics
            epoch_loss_sum += loss.item()
            epoch_recon_loss_sum += recon_loss.item()
            epoch_sparse_loss_sum += sparse_loss.item()
            
            # Backpropagation
            loss.backward()
            
            # Weight normalization and gradient projection
            model.scale_to_unit_norm()
            model.project_grads_decode()
            
            # Gradient clipping for stability
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.clip_grad)
            
            # Update model parameters
            optimizer.step()
            
            # Log metrics periodically
            if global_step % 100 == 0:
                
                # Detach tensors for metric calculation
                recons_sparse, recons_all, embeddings = recons_sparse.detach(), recons_all.detach(), embeddings.detach()
                
                # Calculate evaluation metrics
                # CKNNA (Centered Kernel Nearest Neighbor Alignment) scores
                cknna_score_sparse = cknna(recons_sparse, embeddings)
                cknna_score_all = cknna(recons_all, embeddings)
                
                # FVU (Explained Variance) metric
                fvu_score_sparse = explained_variance(embeddings, recons_sparse)
                fvu_score_all = explained_variance(embeddings, recons_all)
                
                # Reconstruction quality metrics
                diagonal_cs_sparse, mae_distance_sparse = calculate_similarity_metrics(recons_sparse, embeddings)
                diagonal_cs_all, mae_distance_all = calculate_similarity_metrics(recons_all, embeddings)
                
                # Orthogonality of decoder features
                od = orthogonal_decoder(model.decoder)
                
                # Sparsity measurements
                sparsity_sparse = (repr_sparse == 0.0).float().mean(axis=-1).mean()
                sparsity_all = (repr_all == 0.0).float().mean(axis=-1).mean()

                # Representation Metrics
                repr_norm = repr_all.norm(dim=-1).mean().item()
                repr_max = repr_all.max(dim=-1).values.mean().item()

                # Check for dead neurons periodically
                if global_step % cfg.training.check_dead == 0:
                    activations = model.get_and_reset_stats()
                    dead_neurons = identify_dead_neurons(activations).numpy().tolist()
                    numb_of_dead_neurons = len(dead_neurons)
                    logger.info(f"Number of dead neurons: {numb_of_dead_neurons}")
                    
                    # Log dead neuron info to WandB
                    wandb.log({
                        "train/dead_neurons": numb_of_dead_neurons,
                        "train/dead_neurons_ratio": numb_of_dead_neurons / cfg.model.n_latents,
                        "global_step": global_step
                    })
                
                # Get current learning rate
                current_lr = scheduler.get_last_lr()[0] if scheduler else cfg.training.lr
                
                # Log training metrics to WandB
                wandb.log({
                    "train/loss": loss.item(),
                    "train/reconstruction_loss": recon_loss.item(),
                    "train/sparsity_loss": sparse_loss.item(),
                    "train/fvu_sparse": fvu_score_sparse,
                    "train/fvu_all": fvu_score_all,
                    "train/cknna_sparse": cknna_score_sparse,
                    "train/cknna_all": cknna_score_all,
                    "train/cosine_sim_sparse": diagonal_cs_sparse,
                    "train/mae_sparse": mae_distance_sparse,
                    "train/cosine_sim_all": diagonal_cs_all,
                    "train/mae_all": mae_distance_all,
                    "train/sparsity_sparse": sparsity_sparse.item(),
                    "train/sparsity_all": sparsity_all.item(),
                    "train/orthogonal_decoder": od,
                    "train/repr_norm": repr_norm,
                    "train/repr_max": repr_max,
                    "train/grad_norm": grad_norm.item(),
                    "train/learning_rate": current_lr,
                    "global_step": global_step
                })
                
                logger.info(f"Epoch: {epoch+1}, Step: {step}, Loss: {loss.item():.6f}, Recon Loss: {recon_loss.item():.6f}, Sparse Loss: {sparse_loss.item():.6f}")
                logger.info(f"FVU Sparse: {fvu_score_sparse:.4f}, FVU All: {fvu_score_all:.4f}")
                logger.info(f"CKNNA Sparse: {cknna_score_sparse:.4f}, CKNNA All: {cknna_score_all:.4f}")
                logger.info(f"Cosine Similarity Sparse: {diagonal_cs_sparse:.4f}, MAE Distance Sparse: {mae_distance_sparse:.4f}")
                logger.info(f"Cosine Similarity All: {diagonal_cs_all:.4f}, MAE Distance All: {mae_distance_all:.4f}")
                logger.info(f"Sparsity Sparse: {sparsity_sparse:.4f}, Sparsity All: {sparsity_all:.4f}")
                logger.info(f"Orthogonal Decoder Loss: {od:.6f}, Representation norm {repr_norm:.4f} and max {repr_max:.2f}")
                
        # Calculate epoch averages
        epoch_avg_loss = epoch_loss_sum / epoch_steps
        epoch_avg_recon = epoch_recon_loss_sum / epoch_steps
        epoch_avg_sparse = epoch_sparse_loss_sum / epoch_steps
        
        # Log epoch averages
        wandb.log({
            "epoch/train_loss": epoch_avg_loss,
            "epoch/train_reconstruction_loss": epoch_avg_recon,
            "epoch/train_sparsity_loss": epoch_avg_sparse,
        }, step=epoch+1)
        
        # Evaluate the model on the validation set
        logger.info("Running validation evaluation...")
        val_metrics = eval(model, eval_loader, loss_fn, device, cfg, prefix="val")
        wandb.log(val_metrics, step=epoch+1)
        
        # Check if this is the best model (lowest validation FVU)
        current_val_fvu = val_metrics["val/fvu_sparse"]
        if current_val_fvu < best_val_fvu:
            best_val_fvu = current_val_fvu
            logger.info(f"New best validation FVU: {best_val_fvu:.4f}")
            wandb.run.summary["best_val_fvu"] = best_val_fvu
            wandb.run.summary["best_epoch"] = epoch + 1
        
        # Update learning rate
        if scheduler:
            scheduler.step()
        
        # Log epoch summary
        lr_rate = scheduler.get_last_lr()[0] if scheduler else cfg.training.lr
        logger.info(f"Epoch: {epoch+1}, Learning Rate: {lr_rate:.6f}, Loss: {epoch_avg_loss:.6f}, Recon Loss: {epoch_avg_recon:.6f}, Sparse Loss: {epoch_avg_sparse:.6f}, Dead neurons: {numb_of_dead_neurons}")
        
        # Evaluate on second modality if provided
        if has_second_modality:
            logger.info("Running second modality evaluation...")
            second_modality_metrics = eval(model, eval_loader_second, loss_fn, device, cfg, prefix="second_modality")
            wandb.log(second_modality_metrics,step=epoch+1)
            #wandb.log(step=epoch+1)
    
    # Save the trained model
    # For Matryoshka models, append the first nesting level to activation name
    model_appendix = ""
    if args.model == "ReLUSAE":
        activation = f"{args.activation}_{str(cfg.loss.sparse_weight).split('.')[1]}"
    else:
        activation = cfg.model.activation
    
    if cfg.model.use_matryoshka:
        activation += f"_{model.nesting_list[0]}"
        if "RW" in args.model:
            model_appendix = "_RW"
        else:
            model_appendix = "_UW"
        
    # Construct filename with key hyperparameters
    model_params = f"{cfg.model.n_latents}_{cfg.model.n_inputs}_{activation}{model_appendix}_{cfg.model.tied}_{cfg.model.normalize}_{cfg.model.latent_soft_cap}"
    dataset_name = args.dataset_train.split("/")[-1].split(".")[0]
    if "BatchTopK" in args.model:
        save_path = f"/BS/disentanglement/work/msae/batchtopk/{model_params}_{dataset_name}.pth"
    else:
        save_path = f"/BS/disentanglement/work/msae/{model_params}_{dataset_name}.pth"
    
    # Save model state and preprocessing parameters
    checkpoint = {
        "model": model.state_dict(),
        "mean_center": train_ds.mean,
        "scaling_factor": train_ds.scaling_factor,
        "target_norm": train_ds.target_norm,
        "config": wandb_config,
        "epoch": cfg.training.epochs,
        "best_val_fvu": best_val_fvu,
    }
    torch.save(checkpoint, save_path)
    
    logger.info(f"Model saved to {save_path}")
    
    # Save model as WandB artifact
    #artifact = wandb.Artifact(
    #    name=f"sae-{model_type_str}-{args.model}-{wandb.run.id}",
    #    type="model",
    #    description=f"Trained {args.model} on {model_type_display} embeddings with expansion factor {args.expansion_factor}",
    #    metadata={
    #        "model_type": args.model,
    #        "backbone": model_type_display,
    #        "expansion_factor": args.expansion_factor,
    #        "n_latents": cfg.model.n_latents,
    #        "best_val_fvu": best_val_fvu,
    #        "total_epochs": cfg.training.epochs,
    #    }
    #)
    #artifact.add_file(save_path)
    #wandb.log_artifact(artifact)
    
    #logger.info("Model artifact logged to WandB")
    
    # Log final summary statistics
    wandb.run.summary.update({
        "final/train_loss": epoch_avg_loss,
        "final/val_fvu_sparse": val_metrics["val/fvu_sparse"],
        "final/val_cknna_sparse": val_metrics["val/cknna_sparse"],
        "final/dead_neurons": numb_of_dead_neurons,
        "final/dead_neurons_ratio": numb_of_dead_neurons / cfg.model.n_latents,
    })
    
    # Finish WandB run
    wandb.finish()
    logger.info("Training completed!")



def tc_main(args):
    """
    Main training function for Sparse Autoencoders with WandB integration.
    
    This function handles the complete training pipeline:
    1. Setting up configuration based on model type and arguments
    2. Loading and preparing datasets
    3. Initializing the appropriate model (standard or Matryoshka SAE)
    4. Setting up loss function, optimizer, and learning rate scheduler
    5. Executing the training loop with periodic evaluation
    6. Tracking metrics including reconstruction quality, sparsity, and dead neurons
    7. Saving the trained model with relevant metadata
    8. Logging all metrics and artifacts to Weights & Biases
    
    Args:
        args (argparse.Namespace): Command line arguments from parse_args()
    """
    logger.info("Starting training with the following arguments:")
    logger.info(args)
    
    # Get configuration based on model type
    cfg = get_config(args.model)
    cfg.training.epochs = args.epochs
    cfg.training.batch_size = args.batch_size
    cfg.training.num_workers = 4
    cfg.training.lr = float(args.lr)
    #cfg.training.mean_center = False
    ##cfg.training.mean_center=True#cfg.training.mean_center, 
    ##cfg.training.target_norm=None#cfg.training.target_norm,
    
    # Set the random seed for reproducibility
    set_seed(79677) #V1: 68779 V2: 79677
    
    # Set the device (GPU/CPU)
    device = get_device()
    logger.info(f"Using device: {device}")
    
    # Extract model type from dataset path
    model_type_str = None
    if "resnet50" in args.dataset_train:
        model_type_str = "resnet50"
    elif "vit_b_16" in args.dataset_train:
        model_type_str = "vit_b_16"
    elif "dinov2_vitb14" in args.dataset_train:
        model_type_str = "dinov2_vitb14"
    elif "dinov2_vitl14" in args.dataset_train:
        model_type_str = "dinov2_vitl14"
    
    model_type_display = extract_model_type(args.dataset_train)
    
    # Load datasets
    # TODO: add args.dataset_train_input args.dataset_val_input and mean_center
    train_ds_input = SAEDataset(
        args.dataset_train + "_tc_input", 
        dtype=cfg.training.dtype, 
        mean_center=cfg.training.mean_center, 
        target_norm=cfg.training.target_norm,
        split="train",
        model_type=model_type_str
    )
    
    train_ds_output = SAEDataset(
        args.dataset_train + "_tc_output", 
        dtype=cfg.training.dtype, 
        mean_center=cfg.training.mean_center, 
        target_norm=cfg.training.target_norm,
        split="train",
        model_type=model_type_str
    )
    
    eval_ds_input = SAEDataset(
        args.dataset_test + "_tc_input", 
        dtype=cfg.training.dtype, 
        mean_center=cfg.training.mean_center,
        target_norm=cfg.training.target_norm,
        split="val",
        model_type=model_type_str
    )
    
    eval_ds_output = SAEDataset(
        args.dataset_test + "_tc_output", 
        dtype=cfg.training.dtype, 
        mean_center=cfg.training.mean_center,
        target_norm=cfg.training.target_norm,
        split="val",
        model_type=model_type_str
    )
    
    logger.info(f"Training (input) dataset length: {len(train_ds_input)}, Training (output) dataset length: {len(train_ds_output)}, Evaluation (input) dataset length: {len(eval_ds_input)}, Evaluation (output) dataset length: {len(eval_ds_output)}, (Input) Embedding size: {train_ds_input.vector_size}, (Output) Embedding size: {train_ds_output.vector_size}")
    logger.info(f"Training (input) dataset mean center: {train_ds_input.mean.mean()}, Scaling factor: {train_ds_input.scaling_factor} with target norm {train_ds_input.target_norm}")
    logger.info(f"Training (input) dataset mean center: {train_ds_output.mean.mean()}, Scaling factor: {train_ds_output.scaling_factor} with target norm {train_ds_output.target_norm}")
    logger.info(f"Evaluation (output) dataset mean center: {eval_ds_input.mean.mean()}, Scaling factor: {eval_ds_input.scaling_factor} with target norm {eval_ds_input.target_norm}")
    logger.info(f"Evaluation (output) dataset mean center: {eval_ds_output.mean.mean()}, Scaling factor: {eval_ds_output.scaling_factor} with target norm {eval_ds_output.target_norm}")
    
    assert train_ds_input.vector_size == eval_ds_input.vector_size, "Training and evaluation input datasets must have the same embedding size"
    assert train_ds_output.vector_size == eval_ds_output.vector_size, "Training and evaluation output datasets must have the same embedding size"
    
    has_second_modality = False
    if args.dataset_second_modality is not None:
        has_second_modality = True
        eval_ds_second = SAEDataset(
            args.dataset_second_modality,
            dtype=cfg.training.dtype,
            mean_center=cfg.training.mean_center,
            target_norm=cfg.training.target_norm
        )
        logger.info(f"Second modality dataset mean center: {eval_ds_second.mean.mean()}, Scaling factor: {eval_ds_second.scaling_factor} with target norm {eval_ds_second.target_norm}")
        assert train_ds_input.vector_size == eval_ds_second.vector_size, "Training and second modality datasets must have the same embedding size"
    
    # Set model parameters based on dataset and arguments
    # TODO: Add cfg.model.n_outputs
    cfg.model.n_inputs = train_ds_input.vector_size
    cfg.model.n_outputs = train_ds_output.vector_size
    
    # Calculate number of latent dimensions using expansion factor
    cfg.model.n_latents = int(args.expansion_factor * train_ds_input.vector_size)
    logger.info(f"Expansion factor: {args.expansion_factor}, Latent dimensions: {cfg.model.n_latents}")
    
    # Extract l1 from ReLU if applied
    if args.model == "ReLUSAE" and "_" in args.activation:
        args.activation, sparse_weight = args.activation.split("_")
        cfg.loss.sparse_weight = float(f"0.{sparse_weight}")
        logger.info(f"Changing sparsity weight value to {cfg.loss.sparse_weight}")
        
    # Override activation if specified in arguments
    if args.activation:
        cfg.model.activation = args.activation
    
    # Configure Matryoshka SAE parameters if applicable
    if cfg.model.use_matryoshka:
        # Max nesting list
        if cfg.model.nesting_list > cfg.model.max_nesting:
            max_nesting = cfg.model.n_latents
        else:
            max_nesting = cfg.model.max_nesting
        
        # Generate nesting list if a single value was provided
        if isinstance(cfg.model.nesting_list, int):
            logger.info(f"Generating nesting list from {cfg.model.nesting_list} to {max_nesting}")
            start = [cfg.model.nesting_list]
            while start[-1] < max_nesting:
                new_k = start[-1] * 2
                if new_k > max_nesting:
                    break
                start.append(new_k)
            
            if max_nesting not in start:
                start.append(max_nesting)
            cfg.model.nesting_list = start
        
        # Set importance weights for different nesting levels
        if cfg.model.relative_importance == "RW":
            # Reverse weighting - higher weight for larger k values
            cfg.model.relative_importance = list(reversed(list(range(1, len(cfg.model.nesting_list)+1))))
        elif cfg.model.relative_importance == "UW":
            # Uniform weighting - equal weight for all k values
            cfg.model.relative_importance = [1.0] * len(cfg.model.nesting_list)

        logger.info(f"Using Matryoshka with nesting list: {cfg.model.nesting_list} and weighting function: {cfg.model.relative_importance}")
    else:
        logger.info(f"Using standard SAE with {cfg.model.activation} activation")
    
    # Initialize Weights & Biases
    wandb_run_name = create_wandb_run_name(cfg, args, model_type_display)
    
    # Prepare WandB config
    wandb_config = {
        # Model architecture
        "model_type": args.model,
        "backbone": model_type_display,
        "activation": cfg.model.activation,
        "expansion_factor": args.expansion_factor,
        
        "n_inputs": cfg.model.n_inputs,
        "n_outputs": cfg.model.n_outputs,
        
        "n_latents": cfg.model.n_latents,
        "use_matryoshka": cfg.model.use_matryoshka,
        "tied_weights": cfg.model.tied,
        "normalize_decoder": cfg.model.normalize,
        "latent_soft_cap": cfg.model.latent_soft_cap,
        
        # Training hyperparameters
        "epochs": cfg.training.epochs,
        "batch_size": cfg.training.batch_size,
        "learning_rate": cfg.training.lr,
        "weight_decay": cfg.training.weight_decay,
        "beta1": cfg.training.beta1,
        "beta2": cfg.training.beta2,
        "eps": cfg.training.eps,
        "scheduler": cfg.training.scheduler,
        "clip_grad": cfg.training.clip_grad,
        
        # Loss configuration
        "reconstruction_loss": cfg.loss.reconstruction_loss,
        "sparse_loss": cfg.loss.sparse_loss,
        "sparse_weight": cfg.loss.sparse_weight,
        
        # Data preprocessing
        "mean_center": cfg.training.mean_center,
        "target_norm": cfg.training.target_norm,
                
        "bias_init_median": cfg.training.bias_init_median,
        
        # Dataset info
        "train_dataset": os.path.basename(args.dataset_train),
        "eval_dataset": os.path.basename(args.dataset_test),
        
        "input_train_size": len(train_ds_input),
        "input_eval_size": len(eval_ds_input),
        "output_train_size": len(train_ds_output),
        "output_eval_size": len(eval_ds_output),
        "has_second_modality": has_second_modality,
        
        # Device
        "device": str(device),
        "seed": 68779,
    }
    
    # Add Matryoshka-specific config
    if cfg.model.use_matryoshka:
        wandb_config.update({
            "nesting_list": cfg.model.nesting_list,
            "relative_importance": cfg.model.relative_importance,
            "matryoshka_type": "RW" if "RW" in args.model else "UW",
        })
    
    # Prepare tags
    tags = [
        args.model,
        model_type_display,
        cfg.model.activation,
        f"EF{args.expansion_factor}",
    ]
    if cfg.model.use_matryoshka:
        tags.append("Matryoshka")
        tags.append("RW" if "RW" in args.model else "UW")
    tags.extend(args.wandb_tags)
    
    # Initialize WandB run
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=wandb_run_name,
        config=wandb_config,
        tags=tags,
        group=model_type_display,  # Group by backbone model
        mode=args.wandb_mode,
    )
    
    logger.info(f"WandB run initialized: {wandb.run.name} (ID: {wandb.run.id})")
    logger.info(f"WandB URL: {wandb.run.url}")
    
    # Calculate bias initialization (median or zero)
    logger.info(f"Calculating bias initialization with median: {cfg.training.bias_init_median}")
    bias_init_input = 0.0
    bias_init_output = 0.0
    if cfg.training.bias_init_median:
        # Use geometric median of a subset of data points for robustness
        bias_init_input = geometric_median(train_ds_input, device=device, max_number=len(train_ds_input)//10)
        bias_init_output = geometric_median(train_ds_output, device=device, max_number=len(train_ds_output)//10)
    logger.info(f"Bias initialization: {bias_init_input} & {bias_init_output}")
    
    # Log bias initialization to WandB
    wandb.log({"initialization/input_bias_init_norm": torch.norm(bias_init_input).item() if isinstance(bias_init_input, torch.Tensor) else abs(bias_init_input)})
    wandb.log({"initialization/output_bias_init_norm": torch.norm(bias_init_output).item() if isinstance(bias_init_output, torch.Tensor) else abs(bias_init_output)})
    
    # Initialize the appropriate model type
    if cfg.model.use_matryoshka:
        raise NotImplementedError("NO MATRYOSHKA IMPLEMENTATION FOR TRANSCODERS")
        model = MatryoshkaAutoencoder(bias_init=bias_init, **asdict(cfg.model))
    else:
        print(asdict(cfg.model))
        model = Transcoder(input_bias_init=bias_init_input, output_bias_init=bias_init_output, **asdict(cfg.model))
    model = model.to(device)
    
    # Log model architecture
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    wandb.log({
        "model/total_params": total_params,
        "model/trainable_params": trainable_params,
        "model/total_params_millions": total_params / 1e6,
    })
    
    # Prepare loss function
    # Use zeros or calculate mean from dataset depending on config
    input_mean_input = torch.zeros((train_ds_input.vector_size,), dtype=cfg.training.dtype)
    output_mean_input = torch.zeros((train_ds_output.vector_size,), dtype=cfg.training.dtype)
    if not cfg.training.mean_center:
        input_mean_input = calculate_vector_mean(train_ds_input, num_workers=cfg.training.num_workers)
        output_mean_input = calculate_vector_mean(train_ds_output, num_workers=cfg.training.num_workers)
    
    input_mean_input = input_mean_input.to(device)
    output_mean_input = output_mean_input.to(device)
    loss_fn = TCLoss(
        reconstruction_loss=cfg.loss.reconstruction_loss,
        sparse_loss=cfg.loss.sparse_loss,
        sparse_weight=cfg.loss.sparse_weight,
        #input_mean_input=input_mean_input,
        mean_input=output_mean_input
    )
    
    # Prepare the optimizer with adaptive settings based on device
    fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and "cuda" in device.type
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=cfg.training.lr, 
        betas=(cfg.training.beta1, cfg.training.beta2), 
        eps=cfg.training.eps, 
        weight_decay=cfg.training.weight_decay, 
        fused=use_fused
    )
    
    logger.info(f"Using fused AdamW: {use_fused}")
    
    # Prepare the learning rate scheduler
    if cfg.training.scheduler == 1:
        # Linear decay scheduler
        scheduler = LinearDecayLR(optimizer, cfg.training.epochs, decay_time=cfg.training.decay_time)
    elif cfg.training.scheduler == 2:
        # Cosine annealing with warmup
        scheduler = CosineWarmupScheduler(
            optimizer, 
            max_lr=cfg.training.lr, 
            warmup_epoch=1, 
            final_lr_factor=0.1, 
            total_epochs=cfg.training.epochs
        )
    else:
        # No scheduler
        scheduler = None
    
    # Prepare the dataloaders
    train_loader_input = torch.utils.data.DataLoader(
        train_ds_input, 
        batch_size=cfg.training.batch_size, 
        num_workers=cfg.training.num_workers, 
        shuffle=True
    )
    eval_loader_input = torch.utils.data.DataLoader(
        eval_ds_input, 
        batch_size=cfg.training.batch_size, 
        num_workers=cfg.training.num_workers, 
        shuffle=False
    )
    
    train_loader_output = torch.utils.data.DataLoader(
        train_ds_output, 
        batch_size=cfg.training.batch_size, 
        num_workers=cfg.training.num_workers, 
        shuffle=True
    )
    eval_loader_output = torch.utils.data.DataLoader(
        eval_ds_output, 
        batch_size=cfg.training.batch_size, 
        num_workers=cfg.training.num_workers, 
        shuffle=False
    )
    
    if has_second_modality:
        eval_loader_second = torch.utils.data.DataLoader(
            eval_ds_second,
            batch_size=cfg.training.batch_size,
            num_workers=cfg.training.num_workers,
            shuffle=False
        )

    # Training loop
    global_step = 0
    numb_of_dead_neurons = 0
    dead_neurons = []
    best_val_fvu = float('inf')  # Track best validation FVU for model saving
    
    for epoch in range(cfg.training.epochs):
        model.train()
        logger.info(f"Epoch {epoch+1}/{cfg.training.epochs}")
        
        # Epoch metrics accumulators
        epoch_loss_sum = 0.0
        epoch_recon_loss_sum = 0.0
        epoch_sparse_loss_sum = 0.0
        epoch_steps = 0
        
        # Training loop for current epoch
        for step, (input_embeddings, outputs_embeddings) in enumerate(tqdm(zip(train_loader_input, train_loader_output), desc="Training")):
            optimizer.zero_grad()
            global_step += 1
            epoch_steps += 1
            input_embeddings = input_embeddings.to(device)
            outputs_embeddings = outputs_embeddings.to(device)
            
            # Forward pass through model
            recons_sparse, repr_sparse, recons_all, repr_all = model(input_embeddings)
            
            # Compute loss based on model type
            if cfg.model.use_matryoshka:
                raise NotImplementedError("NO MATRYOSHKA IMPLEMENTATION FOR TRANSCODERS")
                # For Matryoshka models, compute weighted loss across all nesting levels
                sparse_loss = loss_fn(recons_all, embeddings, repr_all)[-1]
                recon_loss = loss_fn(recons_sparse[0], embeddings, repr_all)[1]
                
                # Weight reconstruction losses by relative importance
                loss_recon_all = torch.tensor(0., requires_grad=True, device=device)
                for i in range(len(recons_sparse)):
                    current_loss = loss_fn(recons_sparse[i], embeddings, repr_sparse[i])[1]
                    loss_recon_all = loss_recon_all + current_loss * model.relative_importance[i]

                # Normalize by sum of weights
                loss = loss_recon_all / sum(model.relative_importance)
                
                # Use first nesting level for metrics
                repr_sparse = repr_sparse[0]
                recons_sparse = recons_sparse[0]
            else:
                # Standard SAE loss computation
                loss, recon_loss, sparse_loss = loss_fn(recons_sparse, outputs_embeddings, repr_sparse)
            
            # Accumulate epoch metrics
            epoch_loss_sum += loss.item()
            epoch_recon_loss_sum += recon_loss.item()
            epoch_sparse_loss_sum += sparse_loss.item()
            
            # Backpropagation
            loss.backward()
            
            # Weight normalization and gradient projection
            model.scale_to_unit_norm()
            model.project_grads_decode()
            
            # Gradient clipping for stability
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.clip_grad)
            
            # Update model parameters
            optimizer.step()
            
            # Log metrics periodically
            if global_step % 100 == 0:
                
                # Detach tensors for metric calculation
                recons_sparse, recons_all, input_embeddings, outputs_embeddings = recons_sparse.detach(), recons_all.detach(), input_embeddings.detach(), outputs_embeddings.detach()
                
                # Calculate evaluation metrics
                # CKNNA (Centered Kernel Nearest Neighbor Alignment) scores
                cknna_score_sparse = cknna(recons_sparse, outputs_embeddings)
                cknna_score_all = cknna(recons_all, outputs_embeddings)
                
                # FVU (Explained Variance) metric
                fvu_score_sparse = explained_variance(outputs_embeddings, recons_sparse)
                fvu_score_all = explained_variance(outputs_embeddings, recons_all)
                
                # Reconstruction quality metrics
                diagonal_cs_sparse, mae_distance_sparse = calculate_similarity_metrics(recons_sparse, outputs_embeddings)
                diagonal_cs_all, mae_distance_all = calculate_similarity_metrics(recons_all, outputs_embeddings)
                
                # Orthogonality of decoder features
                od = orthogonal_decoder(model.decoder)
                
                # Sparsity measurements
                sparsity_sparse = (repr_sparse == 0.0).float().mean(axis=-1).mean()
                sparsity_all = (repr_all == 0.0).float().mean(axis=-1).mean()

                # Representation Metrics
                repr_norm = repr_all.norm(dim=-1).mean().item()
                repr_max = repr_all.max(dim=-1).values.mean().item()

                # Check for dead neurons periodically
                if global_step % cfg.training.check_dead == 0:
                    activations = model.get_and_reset_stats()
                    dead_neurons = identify_dead_neurons(activations).numpy().tolist()
                    numb_of_dead_neurons = len(dead_neurons)
                    logger.info(f"Number of dead neurons: {numb_of_dead_neurons}")
                    
                    # Log dead neuron info to WandB
                    wandb.log({
                        "train/dead_neurons": numb_of_dead_neurons,
                        "train/dead_neurons_ratio": numb_of_dead_neurons / cfg.model.n_latents,
                        "global_step": global_step
                    })
                
                # Get current learning rate
                current_lr = scheduler.get_last_lr()[0] if scheduler else cfg.training.lr
                
                # Log training metrics to WandB
                wandb.log({
                    "train/loss": loss.item(),
                    "train/reconstruction_loss": recon_loss.item(),
                    "train/sparsity_loss": sparse_loss.item(),
                    "train/fvu_sparse": fvu_score_sparse,
                    "train/fvu_all": fvu_score_all,
                    "train/cknna_sparse": cknna_score_sparse,
                    "train/cknna_all": cknna_score_all,
                    "train/cosine_sim_sparse": diagonal_cs_sparse,
                    "train/mae_sparse": mae_distance_sparse,
                    "train/cosine_sim_all": diagonal_cs_all,
                    "train/mae_all": mae_distance_all,
                    "train/sparsity_sparse": sparsity_sparse.item(),
                    "train/sparsity_all": sparsity_all.item(),
                    "train/orthogonal_decoder": od,
                    "train/repr_norm": repr_norm,
                    "train/repr_max": repr_max,
                    "train/grad_norm": grad_norm.item(),
                    "train/learning_rate": current_lr,
                    "global_step": global_step
                })
                
                logger.info(f"Epoch: {epoch+1}, Step: {step}, Loss: {loss.item():.6f}, Recon Loss: {recon_loss.item():.6f}, Sparse Loss: {sparse_loss.item():.6f}")
                logger.info(f"FVU Sparse: {fvu_score_sparse:.4f}, FVU All: {fvu_score_all:.4f}")
                logger.info(f"CKNNA Sparse: {cknna_score_sparse:.4f}, CKNNA All: {cknna_score_all:.4f}")
                logger.info(f"Cosine Similarity Sparse: {diagonal_cs_sparse:.4f}, MAE Distance Sparse: {mae_distance_sparse:.4f}")
                logger.info(f"Cosine Similarity All: {diagonal_cs_all:.4f}, MAE Distance All: {mae_distance_all:.4f}")
                logger.info(f"Sparsity Sparse: {sparsity_sparse:.4f}, Sparsity All: {sparsity_all:.4f}")
                logger.info(f"Orthogonal Decoder Loss: {od:.6f}, Representation norm {repr_norm:.4f} and max {repr_max:.2f}")
                
        # Calculate epoch averages
        epoch_avg_loss = epoch_loss_sum / epoch_steps
        epoch_avg_recon = epoch_recon_loss_sum / epoch_steps
        epoch_avg_sparse = epoch_sparse_loss_sum / epoch_steps
        
        # Log epoch averages
        wandb.log({
            "epoch/train_loss": epoch_avg_loss,
            "epoch/train_reconstruction_loss": epoch_avg_recon,
            "epoch/train_sparsity_loss": epoch_avg_sparse,
        }, step=epoch+1)
        
        # Evaluate the model on the validation set
        logger.info("Running validation evaluation...")
        val_metrics = eval_tc(model, eval_loader_input, eval_loader_output, loss_fn, device, cfg, prefix="val")
        wandb.log(val_metrics, step=epoch+1)
        
        # Check if this is the best model (lowest validation FVU)
        current_val_fvu = val_metrics["val/fvu_sparse"]
        if current_val_fvu < best_val_fvu:
            best_val_fvu = current_val_fvu
            logger.info(f"New best validation FVU: {best_val_fvu:.4f}")
            wandb.run.summary["best_val_fvu"] = best_val_fvu
            wandb.run.summary["best_epoch"] = epoch + 1
        
        # Update learning rate
        if scheduler:
            scheduler.step()
        
        # Log epoch summary
        lr_rate = scheduler.get_last_lr()[0] if scheduler else cfg.training.lr
        logger.info(f"Epoch: {epoch+1}, Learning Rate: {lr_rate:.6f}, Loss: {epoch_avg_loss:.6f}, Recon Loss: {epoch_avg_recon:.6f}, Sparse Loss: {epoch_avg_sparse:.6f}, Dead neurons: {numb_of_dead_neurons}")
        
        # Evaluate on second modality if provided
        if has_second_modality:
            logger.info("Running second modality evaluation...")
            second_modality_metrics = eval(model, eval_loader_second, loss_fn, device, cfg, prefix="second_modality")
            wandb.log(second_modality_metrics,step=epoch+1)
            #wandb.log(step=epoch+1)
    
    # Save the trained model
    # For Matryoshka models, append the first nesting level to activation name
    model_appendix = ""
    if args.model == "ReLUSAE":
        activation = f"{args.activation}_{str(cfg.loss.sparse_weight).split('.')[1]}"
    else:
        activation = cfg.model.activation
    
    if cfg.model.use_matryoshka:
        activation += f"_{model.nesting_list[0]}"
        if "RW" in args.model:
            model_appendix = "_RW"
        else:
            model_appendix = "_UW"
        
    # Construct filename with key hyperparameters
    model_params = f"{cfg.model.n_latents}_{cfg.model.n_inputs}_{cfg.model.n_outputs}_{activation}{model_appendix}_{cfg.model.tied}_{cfg.model.normalize}_{cfg.model.latent_soft_cap}"
    dataset_name = args.dataset_train.split("/")[-1].split(".")[0]
    if "BatchTopK" in args.model:
        save_path = f"/BS/disentanglement/work/msae/batchtopk/{model_params}_{dataset_name}.pth"
    else:
        save_path = f"/BS/disentanglement/work/msae/{model_params}_{dataset_name}.pth"
    
    # Save model state and preprocessing parameters
    checkpoint = {
        "model": model.state_dict(),
        
        "mean_center": train_ds_input.mean,
        "input_scaling_factor": train_ds_input.scaling_factor,
        "target_norm": train_ds_input.target_norm,
        
        "mean_center": train_ds_output.mean,
        "output_scaling_factor": train_ds_output.scaling_factor,
        "target_norm": train_ds_output.target_norm,
        
        "config": wandb_config,
        "epoch": cfg.training.epochs,
        "best_val_fvu": best_val_fvu,
    }
    torch.save(checkpoint, save_path)
    
    logger.info(f"Model saved to {save_path}")
        
    # Log final summary statistics
    wandb.run.summary.update({
        "final/train_loss": epoch_avg_loss,
        "final/val_fvu_sparse": val_metrics["val/fvu_sparse"],
        "final/val_cknna_sparse": val_metrics["val/cknna_sparse"],
        "final/dead_neurons": numb_of_dead_neurons,
        "final/dead_neurons_ratio": numb_of_dead_neurons / cfg.model.n_latents,
    })
        
    # Finish WandB run
    wandb.finish()
    logger.info("Training completed!")



import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

def create_data_loaders(root: str, batch_size: int = 256, num_workers: int = 4, image_size: int = 224):
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


from metrics import (
    explained_variance_full,
    normalized_mean_absolute_error,
    l0_messure,
    cknna
)
import numpy as np
import torch.nn as nn


def test_feature_similarity(model, target_layer, dataloader, sae=None, device='cuda:0'):
    """
    Compute feature similarity metrics:
    - Intra-class compactness (lower is better): how tight each class is
    - Inter-class separability (higher is better): how far apart classes are
    - Silhouette score (higher is better): ratio of inter to intra distances
    """
    model.eval()
    activations = {}
    
    def hook_fn(module, input, output):
        activations['layer'] = output.detach()
    
    layer = get_layer_by_path(model, target_layer)
    handle = layer.register_forward_hook(hook_fn)
    
    # Collect activations per class
    class_activations = {}
    
    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Collecting class activations"):
            images = images.to(device)
            model(images)
            
            acts = activations['layer']
            
            # Handle different architectures
            if acts.dim() == 4:  # CNN
                acts = acts.sum(dim=(2, 3))
            elif acts.dim() == 3:  # ViT
                acts = acts[:, 0, :]
            
            # Apply SAE if provided
            if sae:
                acts, _ = sae(acts)
            
            acts = acts.cpu()
            
            # Group by class
            for act, target in zip(acts, targets):
                class_id = target.item()
                if class_id not in class_activations:
                    class_activations[class_id] = []
                class_activations[class_id].append(act)
    
    handle.remove()
    
    # Compute statistics
    all_acts = []
    class_means = []
    class_stds = []
    
    for class_id, acts in class_activations.items():
        stacked = torch.stack(acts)
        all_acts.append(stacked)
        class_means.append(stacked.mean(dim=0))
        class_stds.append(stacked.std(dim=0).mean().item())
    
    all_activations = torch.cat(all_acts, dim=0)
    global_std = all_activations.std(dim=0).mean().item()
    
    # Intra-class compactness (normalized)
    avg_within_class_std = np.mean(class_stds)
    intra_class_compactness = avg_within_class_std / global_std
    
    # Inter-class separability
    inter_class_distances = []
    for i in tqdm(range(len(class_means)), desc="Computing inter-class distances"):
        for j in range(i + 1, len(class_means)):
            dist = torch.norm(class_means[i] - class_means[j]).item()
            inter_class_distances.append(dist)
    
    avg_inter_class_distance = np.mean(inter_class_distances)
    inter_class_separability = avg_inter_class_distance / global_std
    
    # Silhouette-like score
    silhouette_score = avg_inter_class_distance / avg_within_class_std
    
    return intra_class_compactness, inter_class_separability, silhouette_score




def hc_eval(model_name, model_orig, model, layer_of_interest, eval_loader, device):
    # Evaluation phase
    local_cknna_score_sparse = []
    global_cknna_score_sparse = []
    fvu_score_sparse = []
    diagonal_cs_sparse = []
    mae_distance_sparse = []
    od = []
    sparsity_sparse = []
    local_hidden_cknna_score_sparse = []
    global_hidden_cknna_score_sparse = []
    
    orig_layer = get_layer_by_path(model_orig, layer_of_interest)
    
    layer_of_interest_dis = layer_of_interest + ".0"
    layer_of_interest_merg = layer_of_interest + ".1"
    layer_dis = get_layer_by_path(model, layer_of_interest_dis)
    layer_merg = get_layer_by_path(model, layer_of_interest_merg)
    
    activations = {}
    def get_activation():
        def hook(model, input, output):
            activations["input"] = input[0].detach()
            activations["output"] = output.detach()
        return hook
    
    handle = layer_merg.register_forward_hook(get_activation())
    orig_handle = orig_layer.register_forward_hook(get_activation())
    # Switch to evaluation mode
    model.eval()
    for images, targets in tqdm(eval_loader, desc="Evaluation"):
        images = images.to(device)
        
        # Forward pass without gradient computation
        with torch.no_grad():
            model(images)
            dis_embedding = activations["input"]
            merg_embedding = activations["output"]
            
            model_orig(images)
            embeddings = activations["output"]
            
            if embeddings.dim() == 4:
                embeddings = embeddings.sum(dim=(2,3))
                dis_embedding = dis_embedding.sum(dim=(2,3))
                merg_embedding = merg_embedding.sum(dim=(2,3))
            elif embeddings.dim() == 3:
                embeddings = embeddings[:, 0, :]
                dis_embedding = dis_embedding[:, 0, :]
                merg_embedding = merg_embedding[:, 0, :]
        
        # Accumulate CKNNA scores
        if images.shape[0] == 256:
            local_cknna_score_sparse.append(cknna(merg_embedding, embeddings, topk=5))
            global_cknna_score_sparse.append(cknna(merg_embedding, embeddings, topk=None))
            
            local_hidden_cknna_score_sparse.append(cknna(dis_embedding, embeddings, topk=5))
            global_hidden_cknna_score_sparse.append(cknna(dis_embedding, embeddings, topk=None))
        
        # Accumulate similarity metrics
        fvu_score_sparse.append(explained_variance_full(embeddings, merg_embedding))
        mae_distance_sparse.append(normalized_mean_absolute_error(embeddings, merg_embedding))
        diagonal_cs_sparse.append((torch.nn.functional.cosine_similarity(embeddings, merg_embedding)))
        sparsity_sparse.append((l0_messure(dis_embedding)))
        
        # Accumulate orthogonality measure
        if isinstance(layer_dis, nn.Conv2d):
            od.append(orthogonal_decoder(layer_dis.weight.view(layer_dis.weight.size(0), -1)))
        else:
            od.append(orthogonal_decoder(layer_dis.weight))
    
    size = dis_embedding.shape[1] / merg_embedding.shape[1] #test_size(layer_dis) + test_size(layer_merg)
    print(dis_embedding.shape, merg_embedding.shape)
    logger.info(f"  Model size: {size:.6f}")

    
    mae_distance_sparse = torch.cat(mae_distance_sparse, dim=0).cpu().numpy()
    diagonal_cs_sparse = torch.cat(diagonal_cs_sparse, dim=0).cpu().numpy()
    sparsity_sparse = torch.cat(sparsity_sparse, dim=0).cpu().numpy()
    fvu_score_sparse = torch.cat(fvu_score_sparse, dim=0).cpu().numpy()
    local_cknna_score_sparse = np.array(local_cknna_score_sparse)
    global_cknna_score_sparse = np.array(global_cknna_score_sparse)
    
    # Log evaluation metrics (averaged over batches)
    logger.info("Evaluation results:")
    logger.info(f"  FVU Sparse: {np.mean(fvu_score_sparse):.4f}")
    logger.info(f"  Local CKNNA Sparse: {np.mean(local_cknna_score_sparse):.4f}")
    logger.info(f"  Global CKNNA Sparse: {np.mean(global_cknna_score_sparse):.4f}")
    logger.info(f"  Cosine Similarity Sparse: {np.mean(diagonal_cs_sparse):.4f}")
    logger.info(f"  MAE Distance Sparse: {np.mean(mae_distance_sparse):.4f}")
    logger.info(f"  Sparsity Sparse: {np.mean(sparsity_sparse):.4f}")
    logger.info(f"  Orthogonal Decoder Loss: {np.mean(od):.6f}")
    
    sae_type = model_name.split("-")[0]
    final_metrics = {
        "model_name": model_name,
        "SAE_TYPE": sae_type,
        #"dead_neurons_pct": round(100 * number_of_dead_neurons / model.n_latents, 2),
        "fvu_mean": float(np.mean(fvu_score_sparse)),
        #"fvu_std": float(np.std(aggregated['sparse_fvu'])),
        "mae_mean": float(np.mean(mae_distance_sparse)),
        #"mae_std": float(np.std(aggregated['sparse_mae'])),
        "cosine_sim_mean": float(np.mean(diagonal_cs_sparse)),
        #"cosine_sim_std": float(np.std(aggregated['sparse_cs'])),
        "local_cknna_mean": float(np.mean(local_cknna_score_sparse)),
        #"local_cknna_std": float(np.std(aggregated['local_sparse_cknnas'])),
        "global_cknna_mean": float(np.mean(global_cknna_score_sparse)),
        
        "local_hidden_cknna_mean": float(np.mean(local_hidden_cknna_score_sparse)),
        #"local_cknna_std": float(np.std(aggregated['local_sparse_cknnas'])),
        "global_hidden_cknna_mean": float(np.mean(global_hidden_cknna_score_sparse)),
        
        #"global_cknna_std": float(np.std(aggregated['global_sparse_cknnas'])),
        "decoder_orthogonality": np.mean(od),
        "expansion_f": size,
    }
    
    wandb.log(final_metrics)
    
    handle.remove()
    orig_handle.remove()
    
    return final_metrics

import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder, CelebA

def get_model(model_name: str, device: str, loss: str):
    """
    Returns a model for image embeddings.
    """
    if model_name.lower() == "resnet50":
        model = torchvision.models.resnet50(pretrained=True)
        embedding_dim = 2048 
        layer_path="layer4.2.conv3"
    elif model_name.lower() == "convnext_tiny":
        model = torchvision.models.convnext_tiny(pretrained=True)
        embedding_dim = 768
    elif model_name.lower() == "vit_b_16":
        model = torchvision.models.vit_b_16(pretrained=True)
        embedding_dim = 768
        layer_path = "encoder.layers.encoder_layer_11.mlp.3"
    elif model_name.lower() == "dinov2_vitb14":
        import torch.hub
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        layer_path="blocks.11.mlp.fc2"
        if loss == "supervised":
            model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_lc')
            #ckpt = torch.load("/BS/disentanglement/work/Disentanglement/dinov2_ckpt/best_checkpoint_vit_large_patch14_dinov2.lvd142m.pth", weights_only=False)["model"]
            #model.load_state_dict(ckpt)
            layer_path="backbone.blocks.11.mlp.fc2"
        embedding_dim = 768
    elif model_name.lower() == "dinov2_vitl14":
        import torch.hub
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
        if loss == "supervised_ft":
            model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_lc')
            ckpt = torch.load("/BS/disentanglement/work/Disentanglement/dinov2_ckpt/best_checkpoint_vit_large_patch14_dinov2.lvd142m.pth", weights_only=False)["model"]
            model.load_state_dict(ckpt)
        embedding_dim =1024
        layer_path="blocks.23.mlp.fc2"
    else:
        raise ValueError(f"Unsupported model {model_name}")
    
    model.to(device).eval()
    return model, layer_path, embedding_dim

def test_size(model) -> float:
    """Return total number of parameters in millions."""
    print("Computing model size...")
    total_params = sum(p.numel() for p in model.parameters())
    return round(total_params / 1_000_000, 1)



import torch
import torch.nn as nn
from pathlib import Path
import numpy as np
from tqdm import tqdm

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


def hc_main(args):
    set_seed(0)
    
    model_type = None
    dis_model = None
    loss = None
    tau = args.tau
    if "resnet50" in args.model_orig:
        model_type = "resnet50"
        dis_model, _ = ut.load_fully_multi_disentangled_resnet(tau=args.tau)
    elif "vit_b_16" in args.model_orig:
        model_type = "vit_b_16"
        dis_model, _ = ut.load_fully_multi_disentangled_vit(tau=args.tau)
    elif "dinov2_vitb14" in args.model_orig:
        splits = args.model_orig.split("-")
        model_type = "dinov2_vitb14" #splits[0] #"dinov2_vitl14"
        loss = splits[-1]
        print("Model:", model_type,", Loss:", loss)
        model_orig2, layer_of_interest, _ = get_model(model_type, "cuda:0", loss=loss)
        dis_model, _ = ut.load_fully_multi_disentangled_dinov2(model_orig2, loss, tau=args.tau)
    else:
        raise NotImplementedError(f"{args.model_orig} is not implemented")
    #elif "dinov2_vitl14" in args.model_orig:
    #    splits = args.model_orig.split("-")
    #    model_type = splits[0] #"dinov2_vitl14"
    #    loss = splits[-1]
    #    print("Model:", model_type,", Loss:", loss)
    #    model_orig2, layer_of_interest, _ = get_model(model_type, "cuda:0", loss=loss)
    #    dis_model, _ = ut.load_fully_multi_disentangled_dinov2(model_orig2, loss, tau=args.tau)
    
    if loss is not None:  
        run_name = f"hc_{model_type}_{loss}-{tau}"
    else:
        run_name = f"hc_{model_type}-{tau}"
        
    model_orig, layer_of_interest, _ = get_model(model_type, "cuda:0", loss=loss)
        
    config = {
        "model_path": run_name,
        "batch_size": args.batch_size,
        "seed": 0,
        "model_type": model_type,
        "tau": tau,
    }
    
    wandb.init(
        project='sae-evaluation',
        group=model_type,
        name=run_name,
        tags=args.wandb_tags + ([model_type] if model_type else []),
        config=config
    )

    dataloader, _ = create_data_loaders("/scratch/inf0/user/mparcham/ILSVRC2012/val_categorized", num_workers=0)
    metrics = hc_eval(run_name, model_orig, dis_model, layer_of_interest, dataloader, "cuda:0")
    from metrics import compute_neuron_clustering_metric
    
    avg_intra, avg_inter, sep_ratio = compute_neuron_clustering_metric(
        model=dis_model,
        target_layer=layer_of_interest + ".0",
        val_loader=dataloader,
        k=100,
        patch_size=64,
        experiment_name=run_name,
        device=get_device(),
        n_workers=4,
        recompute_embeddings=False,
        sae=None,  
    )
    final_metrics = {
        "expansion_f": metrics["expansion_f"],
        "avg_intra": float(avg_intra),
        "avg_inter": float(avg_inter),
        "sep_ratio": float(sep_ratio),
    }
    wandb.log(final_metrics)
    
    wandb.finish()
    
    return

if __name__ == "__main__":
    args = parse_args()
    if not args.train_tc:
        if args.eval_hc:
            hc_main(args)
        else:
            main(args)
    else:
        tc_main(args)