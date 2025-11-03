import torch
import logging
import argparse
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score

from sae import load_model
from utils import SAEDataset, set_seed, get_device

import pandas as pd
import os
import utils as ut

"""
Linear Probe Evaluation for Sparse Autoencoders

This script evaluates how well a Sparse Autoencoder preserves semantic information by:
1. Training a linear classifier (probe) on original data to predict class labels
2. Comparing predictions between original data and its reconstructions
3. Measuring discrepancy using KL divergence and argmax agreement

This approach helps quantify how much semantic/class information is preserved
in the autoencoder's latent space.
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
    Parse command line arguments for the linear probe evaluation.
    
    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description="Evaluate semantic preservation in SAE reconstructions using a linear probe")
    parser.add_argument("-m", "--model", type=str, required=True, 
                       help="Path to the trained SAE model file")
    parser.add_argument("-d", "--data", type=str, required=True, 
                       help="Path to training data embeddings file (.npy)")
    parser.add_argument("-t", "--target", type=str, required=True, 
                       help="Path to training data labels file (.txt)")
    parser.add_argument("-e", "--eval_data", type=str, required=True, 
                       help="Path to evaluation data embeddings file (.npy)")
    parser.add_argument("-o", "--eval_target", type=str, required=True, 
                       help="Path to evaluation data labels file (.txt)")
    parser.add_argument("-b", "--batch-size", type=int, default=512, 
                       help="Batch size for training and evaluation")
    parser.add_argument("-s", "--seed", type=int, default=42, 
                       help="Random seed for reproducibility")
    parser.add_argument("-w", "--num-workers", type=int, default=0, 
                       help="Number of worker processes for data loading")
    parser.add_argument("--tau", type=float, default=0.03, 
                       help="Number of worker processes for data loading")
    parser.add_argument("--eval-hc", type=bool, default=False, 
                       help="Number of worker processes for data loading")

    return parser.parse_args()


def valid_linear_probe(model: torch.nn.Module, linear_probe: torch.nn.Module, 
                      dataset: torch.utils.data.Dataset, device: torch.device, 
                      batch_size: int) -> tuple[list[float], list[float]]:
    """
    Evaluate how well reconstructed data preserves the semantic information in original data.
    
    For each batch of samples:
    1. Gets model reconstructions
    2. Passes both original and reconstructed data through the linear probe
    3. Compares predictions using KL divergence and argmax agreement
    
    Args:
        model (torch.nn.Module): The trained autoencoder model
        linear_probe (torch.nn.Module): The trained linear classifier
        dataset (torch.utils.data.Dataset): Dataset containing the data to evaluate
        device (torch.device): Device to run the evaluation on
        batch_size (int): Batch size for processing
        
    Returns:
        tuple: (kl_divergences, argmax_agreements)
            - kl_divergences: List of KL divergence values between prediction distributions
            - argmax_agreements: List of binary values (1=agree, 0=disagree) for prediction classes
    """
    linear_probe.eval()
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0,
        pin_memory=True
    )

    with torch.no_grad():
        kl = []
        arg_max = []

        for batch in tqdm(dataloader, desc="Evaluating semantic preservation"):
            # Move batch to device
            data = batch.to(device)
            
            # Get reconstruction from model
            _, _, data_reconstruction, _ = model(data)

            # Normalize data (cosine normalization)
            data_norm = data / data.norm(dim=-1, keepdim=True)
            data_reconstruction_norm = data_reconstruction / data_reconstruction.norm(dim=-1, keepdim=True)

            # Get predictions from linear probe
            data_predicted = linear_probe(data_norm)
            data_reconstruction_predicted = linear_probe(data_reconstruction_norm)

            # Apply softmax to get proper probability distributions
            data_predicted_prob = torch.nn.functional.softmax(data_predicted, dim=1)
            data_reconstruction_predicted_prob = torch.nn.functional.softmax(data_reconstruction_predicted, dim=1)

            # Calculate KL divergence for each sample in the batch
            for i in range(data.shape[0]):
                # Extract individual sample predictions
                orig_pred = data_predicted_prob[i:i+1]
                recon_pred = data_reconstruction_predicted_prob[i:i+1]
                
                # KL divergence between the probability distributions
                kl_value = torch.nn.functional.kl_div(
                    recon_pred.log(),
                    orig_pred,
                    reduction='batchmean'
                ).item()
                kl.append(kl_value)
                
                # Check if the predicted class is the same (argmax agreement)
                orig_class = torch.argmax(data_predicted[i])
                recon_class = torch.argmax(data_reconstruction_predicted[i])
                arg_max.append((orig_class == recon_class).item())

    return kl, arg_max


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
    return loader

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


def valid_linear_probe_hc(model, hc_model, linear_probe, dataloader, layer_of_interest, device):
    
    linear_probe.eval()
    
    orig_layer = get_layer_by_path(model, layer_of_interest)
    
    layer_dis = get_layer_by_path(hc_model, layer_of_interest)
    
    activations = {}
    def get_activation():
        def hook(model, input, output):
            #activations["input"] = input[0].detach()
            activations["output"] = output.detach()
        return hook
    
    handle = layer_dis.register_forward_hook(get_activation())
    orig_handle = orig_layer.register_forward_hook(get_activation())
    
    with torch.no_grad():
        kl = []
        arg_max = []
        for images, labels in tqdm(dataloader, desc="Evaluating semantic preservation"):
            images = images.to(device)
            model(images)
            act_orig = activations["output"]
            
            hc_model(images)
            act_recon = activations["output"]
            
            if act_orig.dim() == 4:
                act_orig=act_orig.sum(dim=(2,3))
                act_recon=act_recon.sum(dim=(2,3))
            else:
                act_orig=act_orig[:, 0, :]
                act_recon=act_recon[:, 0, :]
                
                        # Normalize data (cosine normalization)
            data_norm = act_orig / act_orig.norm(dim=-1, keepdim=True)
            data_reconstruction_norm = act_recon / act_recon.norm(dim=-1, keepdim=True)

            # Get predictions from linear probe
            data_predicted = linear_probe(data_norm)
            data_reconstruction_predicted = linear_probe(data_reconstruction_norm)

            # Apply softmax to get proper probability distributions
            data_predicted_prob = torch.nn.functional.softmax(data_predicted, dim=1)
            data_reconstruction_predicted_prob = torch.nn.functional.softmax(data_reconstruction_predicted, dim=1)

            # Calculate KL divergence for each sample in the batch
            for i in range(act_orig.shape[0]):
                # Extract individual sample predictions
                orig_pred = data_predicted_prob[i:i+1]
                recon_pred = data_reconstruction_predicted_prob[i:i+1]
                
                # KL divergence between the probability distributions
                kl_value = torch.nn.functional.kl_div(
                    recon_pred.log(),
                    orig_pred,
                    reduction='batchmean'
                ).item()
                kl.append(kl_value)
                
                # Check if the predicted class is the same (argmax agreement)
                orig_class = torch.argmax(data_predicted[i])
                recon_class = torch.argmax(data_reconstruction_predicted[i])
                arg_max.append((orig_class == recon_class).item())
            
    handle.remove()
    orig_handle.remove()
    
    return kl, arg_max

def evaluate_linear_probe(linear_probe: torch.nn.Module, 
                         dataset: torch.utils.data.Dataset, 
                         device: torch.device) -> float:
    """
    Evaluate the linear probe's performance on a dataset using macro F1 score.
    
    Args:
        linear_probe (torch.nn.Module): The trained linear classifier
        dataset (torch.utils.data.Dataset): Dataset containing data and targets
        device (torch.device): Device to run the evaluation on
        
    Returns:
        float: Macro-averaged F1 score across all classes
    """
    linear_probe.eval()
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=64, shuffle=False, num_workers=0
    )

    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for data, target in tqdm(dataloader, total=len(dataloader), desc="Evaluating classifier"):
            data = data.to(device)
            target = target.to(device)

            # Normalize input data
            data = data / data.norm(dim=-1, keepdim=True)
            
            # Get predictions
            outputs = linear_probe(data)
            _, predicted = torch.max(outputs, 1)
            
            # Collect targets and predictions
            all_targets.extend(target.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    # Calculate F1 score once on all predictions
    return f1_score(all_targets, all_predictions, average="macro")



def train_linear_probe(model_name, model: torch.nn.Module, dataset: torch.utils.data.Dataset, 
                      eval_dataset: torch.utils.data.Dataset, target_path: str, 
                      eval_target_path: str, device: torch.device, 
                      batch_size: int, num_workers: int) -> torch.nn.Module:
    """
    Train a linear classifier (probe) on data embeddings to predict class labels.
    
    This function:
    1. Prepares datasets with class labels
    2. Trains a linear classifier using cross-entropy loss
    3. Implements early stopping based on evaluation F1 score
    4. Returns the best model based on validation performance
    
    Args:
        model (torch.nn.Module): The trained autoencoder model (used for dimensions)
        dataset (torch.utils.data.Dataset): Dataset containing training data
        eval_dataset (torch.utils.data.Dataset): Dataset containing evaluation data
        target_path (str): Path to file containing training data labels
        eval_target_path (str): Path to file containing evaluation data labels
        device (torch.device): Device to run the training on
        batch_size (int): Batch size for training
        num_workers (int): Number of worker processes for data loading
        
    Returns:
        torch.nn.Module: The trained linear probe (classifier)
    """    
    # Create data loaders to efficiently load data in batches
    train_dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    
    # Load and prepare training data
    logger.info("Loading and preparing data...")
    data_list = []
    for batch in tqdm(train_dataloader, desc="Loading training data"):
        data_list.append(batch)
    data = torch.cat(data_list, dim=0)
    
    # Load and prepare evaluation data
    eval_data_list = []
    for batch in tqdm(eval_dataloader, desc="Loading evaluation data"):
        eval_data_list.append(batch)
    eval_data = torch.cat(eval_data_list, dim=0)

    # Load and prepare training targets
    with open(target_path, "r") as f:
        target_dataset = f.readlines()
    target_dataset = [x.strip() for x in target_dataset]

    # Create mapping from text labels to numeric indices
    unique_texts = list(dict.fromkeys(target_dataset))
    unique_target = {text: idx for idx, text in enumerate(unique_texts)}
    logger.info(f"Number of unique labels: {len(unique_target)}")
    
    # Convert text labels to numeric indices
    target = torch.empty(len(target_dataset), dtype=torch.long)
    for idx, label in enumerate(target_dataset):
        target[idx] = unique_target[label]

    # Verify data and target dimensions match
    assert data.shape[0] == len(target), f"Data shape {data.shape} and target shape {target.shape} do not match"

    # Load and prepare evaluation targets
    with open(eval_target_path, "r") as f:
        eval_target_dataset = f.readlines()
    eval_target_dataset = [x.strip() for x in eval_target_dataset]

    # Verify evaluation target labels match training target labels
    assert set(list(dict.fromkeys(eval_target_dataset))) <= set(unique_texts), "Evaluation target labels not found in training target labels"
    
    # Convert evaluation text labels to numeric indices
    eval_target = torch.empty(len(eval_target_dataset), dtype=torch.long)
    for idx, label in enumerate(eval_target_dataset):
        eval_target[idx] = unique_target[label]

    # Create datasets and dataloaders for training
    train_tensor_dataset = torch.utils.data.TensorDataset(data, target)
    train_dataloader = torch.utils.data.DataLoader(
        train_tensor_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    
    eval_tensor_dataset = torch.utils.data.TensorDataset(eval_data, eval_target)
    eval_dataloader = torch.utils.data.DataLoader(
        eval_tensor_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    # Define training hyperparameters
    epochs = 50
    patience = 5  # Number of epochs to wait for improvement
    n_classes = len(unique_target)
    
    # Create linear probe model
    linear_probe = torch.nn.Sequential(
        torch.nn.Linear(model.n_inputs, n_classes),
    )
    linear_probe.to(device)

    # Define optimizer, scheduler and loss function
    optimizer = torch.optim.AdamW(linear_probe.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    criterion = torch.nn.CrossEntropyLoss()
    
    save_path = f"/BS/disentanglement/work/sae/linear_probe/linear_probe_best_{model_name}.pth"
    if os.path.exists(save_path):
        logger.info(f"Model already exists at {save_path}. Loading it instead of training...")
        linear_probe = torch.nn.Sequential(
            torch.nn.Linear(model.n_inputs, len(unique_target)),
        )
        linear_probe.load_state_dict(torch.load(save_path)["model_state_dict"])
        linear_probe.to(device)
        return linear_probe
    else:
        logger.info("No existing model found. Starting training...")

    logger.info(f"Training linear probe with {n_classes} classes for {epochs} epochs")
    best_acc = 0.0
    best_model = None
    patience_counter = 0

    # Training loop
    for epoch in range(epochs):
        # Training phase
        linear_probe.train()
        train_loss = 0.0
        
        for embeddings, labels in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            embeddings = embeddings.to(device)
            labels = labels.to(device)

            # Normalize embeddings
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
            
            # Forward pass
            outputs = linear_probe(embeddings)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_dataloader)

        # Evaluation phase
        acc = evaluate_linear_probe(linear_probe, eval_tensor_dataset, device)
        if scheduler is not None:
            scheduler.step(acc)

        logger.info(f"Epoch: {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Eval F1: {acc:.4f}")

        # Save best model and implement early stopping
        if acc > best_acc:
            best_acc = acc
            best_model = linear_probe.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered. Best F1: {best_acc:.4f}")
                break

    # Load best model
    linear_probe.load_state_dict(best_model)
    logger.info(f"Linear probe training complete with best eval F1: {best_acc:.4f}")
    
    save_path = f"/BS/disentanglement/work/sae/linear_probe/linear_probe_best_{model_name}.pth"
    torch.save({
        "model_state_dict": best_model,
        "label_mapping": unique_target,  # Optional: to keep track of labels
    }, save_path)
    logger.info(f"Saved best model to {save_path}")

    return linear_probe

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

def main(args):
    """
    Main function to run the linear probe evaluation pipeline.
    
    This function:
    1. Loads the trained SAE model
    2. Prepares the datasets
    3. Trains a linear probe classifier
    4. Evaluates semantic preservation between original and reconstructed data
    
    Args:
        args (argparse.Namespace): Command line arguments
    """
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Get the device to use for training
    device = get_device()
    logger.info(f"Using device: {device}")
    
    # Load the trained SAE model
    model, mean_center, scaling_factor, target_norm = load_model(args.model)
    model.to(device)
    model.eval()
    logger.info(f"Model loaded: {args.model}")
    
    model_type = None
    if "resnet50" in args.model:
        model_type = "resnet50"
    elif "vit_b_16" in args.model:
        model_type = "vit_b_16"
    elif "dinov2_vitl14" in args.model:
        model_type = "dinov2_vitl14"
            
    # Load the dataset with appropriate preprocessing
    if ("text" in args.model and "text" in args.data) or ("image" in args.model and "image" in args.data):
        logger.info("Using model mean and scalling factor")    
        train_ds = SAEDataset(args.data,split="train", model_type=model_type)
        train_ds.mean = mean_center.cpu()
        train_ds.scaling_factor = scaling_factor
    else:    
        logger.info("Computing mean and scalling factor")    
        train_ds = SAEDataset(args.data, mean_center=True if mean_center.sum() != 0.0 else False, target_norm=target_norm, split="train", model_type=model_type)
    logger.info(f"Training dataset loaded with {len(train_ds)} samples")
    
    if ("text" in args.model and "text" in args.eval_data) or ("image" in args.model and "image" in args.eval_data):
        logger.info("Using model mean and scalling factor")    
        eval_ds = SAEDataset(args.eval_data, split="val", model_type=model_type)
        eval_ds.mean = mean_center.cpu()
        eval_ds.scaling_factor = scaling_factor
    else:    
        logger.info("Computing mean and scalling factor")    
        eval_ds = SAEDataset(args.eval_data, mean_center=True if mean_center.sum() != 0.0 else False, target_norm=target_norm, split="val", model_type=model_type)
    logger.info(f"Evaluation dataset loaded with {len(eval_ds)} samples")

    # Train the linear probe classifier
    logger.info("Training linear probe classifier...")
    model_name=model_type
    linear_probe = train_linear_probe(
        model_name,
        model, train_ds, eval_ds, args.target, args.eval_target, 
        device, args.batch_size, args.num_workers
    )
    
    # Evaluate semantic preservation
    model_orig, layer_path, _ = get_model(model_type, device)
    
    logger.info("Evaluating semantic preservation in reconstructions...")
    if args.eval_hc:
        dataLoader = create_data_loaders("/scratch/inf0/user/mparcham/ILSVRC2012/val_categorized")
        if model_type == "resnet50":
            hc_model, _ = ut.load_fully_multi_disentangled_resnet(tau=args.tau)
        elif model_type == "vit_b_16":
            hc_model, _ = ut.load_fully_multi_disentangled_vit(tau=args.tau)
        eval_kl, eval_arg_max = valid_linear_probe_hc(model_orig, hc_model, linear_probe, dataLoader, layer_path, device)
                                #model, hc_model, linear_probe, dataloader, layer_of_interest, device
        train_kl, train_arg_max = eval_kl, eval_arg_max
    else:
        train_kl, train_arg_max = valid_linear_probe(model, linear_probe, train_ds, device, args.batch_size)
        eval_kl, eval_arg_max = valid_linear_probe(model, linear_probe, eval_ds, device, args.batch_size)

    # Report results
    logger.info("Results Summary:")
    logger.info("Training Set:")
    logger.info(f"  KL Divergence: {np.mean(train_kl):.4f} ± {np.std(train_kl):.4f}")
    logger.info(f"  Class Prediction Agreement: {np.mean(train_arg_max)*100:.2f}% ± {np.std(train_arg_max)*100:.2f}%")
    
    logger.info("Evaluation Set:")
    logger.info(f"  KL Divergence: {np.mean(eval_kl):.4f} ± {np.std(eval_kl):.4f}")
    logger.info(f"  Class Prediction Agreement: {np.mean(eval_arg_max)*100:.2f}% ± {np.std(eval_arg_max)*100:.2f}%")
    
    results = {
            "Model Name": [args.model],
            "KL Divergence": [np.mean(eval_kl)],
            "Class Prediction Agreement": [np.mean(eval_arg_max)*100]
        }

    df = pd.DataFrame(results)

    # Append instead of overwrite
    csv_file = "/BS/disentanglement/work/Disentanglement/MSAE/results_linear.csv" if not args.eval_hc else "/BS/disentanglement/work/Disentanglement/MSAE/results_linear_hc.csv"
    df.to_csv(csv_file, mode="a", index=False, header=not os.path.exists(csv_file))
    
    logger.info("Interpretation:")
    logger.info("  - Lower KL Divergence indicates better preservation of semantic information")
    logger.info("  - Higher Class Prediction Agreement indicates better preservation of class identity")


if __name__ == "__main__":
    args = parse_args()
    main(args)