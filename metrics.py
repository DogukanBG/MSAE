import torch

import torch.nn.functional as F
def calculate_similarity_metrics(original_matrix: torch.Tensor, reconstruction_matrix: torch.Tensor):
    """
    Calculate cosine similarity and Euclidean distance between original and reconstructed vectors.
    
    Computes the average cosine similarity and Euclidean distance for corresponding pairs of
    vectors in the original and reconstruction matrices.
    
    Args:
        original_matrix (torch.Tensor): Original data matrix of shape [batch_size, feature_dim]
        reconstruction_matrix (torch.Tensor): Reconstructed data matrix of shape [batch_size, feature_dim]
        
    Returns:
        tuple:
            - torch.Tensor: Mean cosine similarity (higher values indicate better reconstruction)
            - torch.Tensor: Mean Euclidean distance (lower values indicate better reconstruction)
    """
    # Calculate cosine similarity for each pair
    # First normalize the vectors
    original_norm = original_matrix.norm(dim=-1, keepdim=True)
    reconstruction_norm = reconstruction_matrix.norm(dim=-1, keepdim=True)
    
    original_normalized = original_matrix / original_norm
    reconstruction_normalized = reconstruction_matrix / reconstruction_norm
    
    # Calculate dot product of normalized vectors
    cosine_similarities = reconstruction_normalized @ original_normalized.T
    cosine_similarities = torch.diagonal(cosine_similarities)
    
    # Calculate Euclidean distance for each pair
    euclidean_distances = torch.norm(original_matrix - reconstruction_matrix, dim=-1)
    
    return torch.mean(cosine_similarities), torch.mean(euclidean_distances)


def identify_dead_neurons(latent_bias: torch.Tensor, threshold: float = 10**(-5.5)) -> torch.Tensor:
    """
    Identify dead neurons based on their bias values.
    
    Dead neurons are those with bias magnitudes below a specified threshold,
    indicating that they may not be activating significantly during training.
    
    Args:
        latent_bias (torch.Tensor): Bias vector for latent neurons
        threshold (float, optional): Threshold below which a neuron is considered dead.
                                     Defaults to 10^(-5.5).
    
    Returns:
        torch.Tensor: Indices of dead neurons
    """
    dead_neurons = torch.where(torch.abs(latent_bias) < threshold)[0]
    return dead_neurons


def hsic_unbiased(K: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
    """
    Compute the unbiased Hilbert-Schmidt Independence Criterion (HSIC).
    
    This implementation follows Equation 5 in Song et al. (2012), which provides
    an unbiased estimator of HSIC. This measure quantifies the dependency between
    two sets of variables represented by their kernel matrices.
    
    Reference: 
        Song, L., Smola, A., Gretton, A., & Borgwardt, K. (2012).
        "A dependence maximization view of clustering."
        https://jmlr.csail.mit.edu/papers/volume13/song12a/song12a.pdf
    
    Args:
        K (torch.Tensor): First kernel matrix of shape [n, n]
        L (torch.Tensor): Second kernel matrix of shape [n, n]
        
    Returns:
        torch.Tensor: Unbiased HSIC value (scalar)
    """
    m = K.shape[0]

    # Zero out the diagonal elements of K and L
    K_tilde = K.clone().fill_diagonal_(0)
    L_tilde = L.clone().fill_diagonal_(0)

    # Compute HSIC using the formula in Equation 5
    HSIC_value = (
        (torch.sum(K_tilde * L_tilde.T))
        + (torch.sum(K_tilde) * torch.sum(L_tilde) / ((m - 1) * (m - 2)))
        - (2 * torch.sum(torch.mm(K_tilde, L_tilde)) / (m - 2))
    )

    HSIC_value /= m * (m - 3)
    return HSIC_value


def hsic_biased(K: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
    """
    Compute the biased Hilbert-Schmidt Independence Criterion (HSIC).
    
    This is the original form used in Centered Kernel Alignment (CKA).
    It's computationally simpler than the unbiased version but may have
    statistical bias, especially for small sample sizes.
    
    Args:
        K (torch.Tensor): First kernel matrix of shape [n, n]
        L (torch.Tensor): Second kernel matrix of shape [n, n]
        
    Returns:
        torch.Tensor: Biased HSIC value (scalar)
    """
    H = torch.eye(K.shape[0], dtype=K.dtype, device=K.device) - 1 / K.shape[0]
    return torch.trace(K @ H @ L @ H)


def cknna(feats_A: torch.Tensor, feats_B: torch.Tensor, topk: int = 10, 
         distance_agnostic: bool = False, unbiased: bool = True) -> float:
    """
    Compute the Centered Kernel Nearest Neighbor Alignment (CKNNA). From:
    https://github.com/minyoungg/platonic-rep/blob/4dd084e1b96804ddd07ae849658fbb69797e319b/metrics.py#L180
    
    CKNNA is a variant of CKA that only considers k-nearest neighbors when computing
    similarity. This makes it more robust to outliers and more sensitive to local
    structure in the data.
    
    Args:
        feats_A (torch.Tensor): First feature matrix of shape [n_samples, n_features_A]
        feats_B (torch.Tensor): Second feature matrix of shape [n_samples, n_features_B]
        topk (int, optional): Number of nearest neighbors to consider. Defaults to 10.
        distance_agnostic (bool, optional): If True, only considers binary neighborhood
                                           membership without weighting by similarity.
                                           Defaults to False.
        unbiased (bool, optional): If True, uses unbiased HSIC estimator. 
                                  Defaults to True.
    
    Returns:
        float: CKNNA similarity score between 0 and 1, where higher values
               indicate greater similarity between the feature spaces
               
    Raises:
        ValueError: If topk is less than 2
    """
    n = feats_A.shape[0]
            
    if topk is None:
        topk = feats_A.shape[0] - 1
        
    if topk < 2:
        raise ValueError("CKNNA requires topk >= 2")
                        
    # Compute kernel matrices (linear kernels)
    K = feats_A @ feats_A.T
    L = feats_B @ feats_B.T
    device = feats_A.device

    def similarity(K, L, topk):
        """
        Compute similarity based on nearest neighbor intersection.
        
        This inner function computes similarity between two kernel matrices
        based on their shared nearest neighbor structure.
        """                    
        if unbiased:            
            # Fill diagonal with -inf to exclude self-similarity when finding topk
            K_hat = K.clone().fill_diagonal_(float("-inf"))
            L_hat = L.clone().fill_diagonal_(float("-inf"))
        else:
            K_hat, L_hat = K, L

        # Get topk indices for each row
        _, topk_K_indices = torch.topk(K_hat, topk, dim=1)
        _, topk_L_indices = torch.topk(L_hat, topk, dim=1)
        
        # Create masks for nearest neighbors
        mask_K = torch.zeros(n, n, device=device).scatter_(1, topk_K_indices, 1)
        mask_L = torch.zeros(n, n, device=device).scatter_(1, topk_L_indices, 1)
        
        # Intersection of nearest neighbors
        mask = mask_K * mask_L
                    
        if distance_agnostic:
            # Simply count shared neighbors without considering similarity values
            sim = mask * 1.0
        else:
            # Compute HSIC on the masked kernel matrices
            if unbiased:
                sim = hsic_unbiased(mask * K, mask * L)
            else:
                sim = hsic_biased(mask * K, mask * L)
        return sim

    # Compute similarities
    sim_kl = similarity(K, L, topk)  # Cross-similarity
    sim_kk = similarity(K, K, topk)  # Self-similarity of K
    sim_ll = similarity(L, L, topk)  # Self-similarity of L
            
    # Normalized similarity (similar to correlation)
    return sim_kl.item() / (torch.sqrt(sim_kk * sim_ll) + 1e-6).item()


def explained_variance_full(
    original_input: torch.Tensor,
    reconstruction: torch.Tensor,
) -> float:
    """
    Computes the explained variance between the original input and its reconstruction.

    The explained variance is a measure of how much of the variance in the original input
    is captured by the reconstruction. It is calculated as:
        1 - (variance of the reconstruction error / total variance of the original input)

    Args:
        original_input (torch.Tensor): The original input tensor.
        reconstruction (torch.Tensor): The reconstructed tensor.

    Returns:
        float: The explained variance score, a value between 0 and 1.
            A value of 1 indicates perfect reconstruction.
    """
    variance = (original_input - reconstruction).var(dim=-1)
    total_variance = original_input.var(dim=-1)
    return variance / total_variance


def explained_variance(
    original_input: torch.Tensor,
    reconstruction: torch.Tensor,
) -> float:
    """
    Computes the explained variance between the original input and its reconstruction.

    The explained variance is a measure of how much of the variance in the original input
    is captured by the reconstruction. It is calculated as:
        1 - (variance of the reconstruction error / total variance of the original input)

    Args:
        original_input (torch.Tensor): The original input tensor.
        reconstruction (torch.Tensor): The reconstructed tensor.

    Returns:
        float: The explained variance score, a value between 0 and 1.
            A value of 1 indicates perfect reconstruction.
    """
    return explained_variance_full(original_input, reconstruction).mean(dim=-1).item()


def orthogonal_decoder(decoder: torch.Tensor) -> float:
    """
    Compute the degree of non-orthogonality in decoder weights.
    
    This metric measures how close the decoder feature vectors are to being
    orthogonal to each other. Lower values indicate more orthogonal features,
    which is often desirable for sparse representation learning.
    
    Args:
        decoder (torch.Tensor): Decoder weight matrix of shape [n_latents, n_inputs]
        
    Returns:
        float: Orthogonality score (lower is better, 0 means perfectly orthogonal)
    """
    # Compute dot products between all pairs of decoder vectors
    logits = decoder @ decoder.T
    
    # Create a mask to only consider off-diagonal elements
    I = 1 - torch.eye(decoder.shape[0], device=decoder.device, dtype=decoder.dtype)
    
    # Compute mean squared dot product (excluding diagonal)
    return ((logits * I) ** 2).mean().item()


def normalized_mean_absolute_error(
    original_input: torch.Tensor,
    reconstruction: torch.Tensor,
) -> torch.Tensor:
    """
    Compute normalized mean absolute error between original and reconstructed data.
    
    This metric normalizes the MAE by the mean absolute value of the original input,
    making it scale-invariant and more comparable across different datasets.
    
    Args:
        original_input (torch.Tensor): Original input data of shape [batch, n_inputs]
        reconstruction (torch.Tensor): Reconstructed data of shape [batch, n_inputs]
        
    Returns:
        torch.Tensor: Normalized MAE for each sample in the batch
    """
    return (
        torch.abs(reconstruction - original_input).mean(dim=1) / 
        torch.abs(original_input).mean(dim=1)
    )


def normalized_mean_squared_error(
    original_input: torch.Tensor,
    reconstruction: torch.Tensor,
) -> torch.Tensor:
    """
    Compute normalized mean squared error between original and reconstructed data.
    
    This metric normalizes the MSE by the mean squared value of the original input,
    making it scale-invariant and more comparable across different datasets.
    Also known as the Fraction of Variance Unexplained (FVU).
    
    Args:
        original_input (torch.Tensor): Original input data of shape [batch, n_inputs]
        reconstruction (torch.Tensor): Reconstructed data of shape [batch, n_inputs]
        
    Returns:
        torch.Tensor: Normalized MSE for each sample in the batch
    """
    return (
        ((reconstruction - original_input) ** 2).mean(dim=1) / 
        (original_input**2).mean(dim=1)
    )


def l0_messure(sample: torch.Tensor) -> torch.Tensor:
    """
    Compute the L0 measure (sparsity) of feature activations.
    
    The L0 measure counts the proportion of zero elements in the activation,
    providing a direct measure of sparsity. Higher values indicate greater
    sparsity (more zeros).
    
    Note: The function name contains a spelling variant ("messure" vs "measure")
    but is kept for backward compatibility.
    
    Args:
        sample (torch.Tensor): Activation tensor of shape [batch, n_features]
        
    Returns:
        torch.Tensor: Proportion of zero elements for each sample in the batch
    """
    return (sample == 0).float().mean(dim=1)

import torch
import torch.nn as nn
from pathlib import Path
import numpy as np
from tqdm import tqdm
import os

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

def monosemanticity_score_concept(experiment_name, model, sae, target_layer, val_loader, feature_extractor):
    """
    hook to capture activations
    """
    if "dinov2_b" == feature_extractor:
        """
            Check if image similarity matrix exists
            if yes: load
            if no: load model, compute and save
            Make sure that val_loader order and similarity matrix order is consistent
        """
    """
    check if embeddings exists:
        if yes: load
        else:
            all_latents = []
            for images, targets in val_loader:
                images.to(device)
                model(images)
                latent_acts = activations["output"]
            if latent_acts.dim() == 4:
                latent_acts = latent_acts.sum(dim=(2,3))
            elif latent_acts.dim() == 3:
                latent_acts = latent_acts[:,0,:]
            if not sae is None:
                latent_acts = sae(latent_acts)
            all_latents.append(latents_acts)
        all_latents = torch.stack(all_latents, dim=0)
        torch.save(f"monosemanticity_score_{experiment_name}.pkl")
        compute monosemanticity score:
            monosemanticity_scores = []
            for each channel k in num_channels
                ms_score_sum = 0
                for n in num_images:
                    act1 = all_latents[n]
                    for m in num_images:
                        if n == m:
                            skip
                        act2 = all_latents[m]
                        r_k_nm = act1[k] * act2[k]
                        s_nm = similarity_matrix[n][m]
                        ms_score_sum += r_k_nm * s_nm
                ms_score = ms_score_sum / (num_images * (num_images-1))
            monosemanticity_scores.append(ms_score)
            ms_score_avg_over_all_channels = sum(monosemanticity_scores) / len(monosemanticity_scores)
            return ms_score_avg_over_all_channels
    
    
    MS^k = 1/(N*(N-1)) * sum(n=1, num_images) * sum(m=1, num_images) r^k_nm * s_nm
    """
    return


import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import pickle
import json
import gc
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_distances
import torchvision.transforms as transforms
import clip


class PatchDataset(Dataset):
    """Dataset of image patches extracted from ImageNet validation set."""
    
    def __init__(self, patches, transform=None):
        self.patches = patches
        self.transform = transform
    
    def __len__(self):
        return len(self.patches)
    
    def __getitem__(self, idx):
        patch = self.patches[idx]
        if self.transform:
            patch = self.transform(patch)
        return patch


def get_layer_by_path(model, layer_path):
    """Get layer from model using dot-notation path."""
    parts = layer_path.split('.')
    layer = model
    for part in parts:
        layer = getattr(layer, part)
    return layer


def extract_patches_from_imagenet(val_loader, patch_size=16, save_dir=None, device='cuda'):
    """
    Extract all patches from ImageNet validation dataset.
    This is model-independent and only needs to be computed once.
    """
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        patch_file = save_dir / f"imagenet_patches_size{patch_size}.pkl"
        
        # Load if exists
        if patch_file.exists():
            print(f"Loading cached patches from {patch_file}")
            with open(patch_file, 'rb') as f:
                data = pickle.load(f)
            return data['patches'], data['metadata']
    
    print("Extracting patches from ImageNet validation set...")
    all_patches = []
    all_metadata = []
    
    for img_idx, (images, _) in enumerate(tqdm(val_loader, desc="Extracting patches")):
        batch_size = images.size(0)
        
        for b in range(batch_size):
            img = images[b]  # (C, H, W)
            _, H, W = img.shape
            
            # Calculate number of patches
            num_patches_h = H // patch_size
            num_patches_w = W // patch_size
            
            # Extract patches
            for i in range(num_patches_h):
                for j in range(num_patches_w):
                    h_start = i * patch_size
                    w_start = j * patch_size
                    patch = img[:, h_start:h_start+patch_size, w_start:w_start+patch_size]
                    
                    # Convert to PIL Image
                    patch_pil = transforms.ToPILImage()(patch)
                    all_patches.append(patch_pil)
                    
                    all_metadata.append({
                        'img_idx': img_idx * batch_size + b,
                        'position': (i, j),
                        'patch_idx': len(all_patches) - 1
                    })
    
    metadata = {
        'patch_size': patch_size,
        'num_patches': len(all_patches),
        'metadata': all_metadata
    }
    
    # Save if directory provided
    if save_dir is not None:
        print(f"Saving patches to {patch_file}")
        with open(patch_file, 'wb') as f:
            pickle.dump({'patches': all_patches, 'metadata': metadata}, f)
    
    return all_patches, metadata


def extract_patch_activations(model, patches, patch_metadata, target_layer, experiment_name,
                              batch_size=64, device='cuda', sae=None):
    """
    Extract activations for all patches through the target layer.
    Returns memmap info dict instead of loading all activations.
    """
    # Create dataset and dataloader
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    patch_dataset = PatchDataset(patches, transform=transform)
    patch_loader = DataLoader(patch_dataset, batch_size=batch_size, 
                             shuffle=False, num_workers=4, pin_memory=True)
    
    # Setup hook
    activations_dict = {}
    def hook_fn(module, input, output):
        activations_dict["output"] = output.detach()
    
    layer = get_layer_by_path(model, target_layer)
    hook_handle = layer.register_forward_hook(hook_fn)
    
    model = model.to(device)
    model.eval()
    
    if sae is not None:
        sae = sae.to(device)
        sae.eval()
    
    # First pass to determine output shape
    first_batch = next(iter(patch_loader)).to(device)
    with torch.no_grad():
        _ = model(first_batch)
        latent_acts = activations_dict["output"]
        
        if latent_acts.dim() == 4:
            latent_acts = latent_acts.mean(dim=(2, 3))
        elif latent_acts.dim() == 3:
            latent_acts = latent_acts[:, 0, :]
        
        if sae is not None:
            _, latent_acts, _, _ = sae(latent_acts)
        
        num_neurons = latent_acts.shape[1]
    
    # Create memory-mapped array
    num_patches = len(patches)
    mmap_file = f'/tmp/patch_activations_{experiment_name}.dat'
    activations_mmap = np.memmap(mmap_file, dtype='float32', mode='w+', 
                                  shape=(num_patches, num_neurons))
    
    current_idx = 0
    
    # Recreate dataloader (first batch was consumed)
    patch_loader = DataLoader(patch_dataset, batch_size=batch_size, 
                             shuffle=False, num_workers=4, pin_memory=True)
    
    with torch.no_grad():
        for patch_batch in tqdm(patch_loader, desc="Computing patch activations"):
            patch_batch = patch_batch.to(device)
            
            # Forward pass
            _ = model(patch_batch)
            latent_acts = activations_dict["output"]
            
            # Handle different activation shapes
            if latent_acts.dim() == 4:
                latent_acts = latent_acts.mean(dim=(2, 3))
            elif latent_acts.dim() == 3:
                latent_acts = latent_acts[:, 0, :]
            
            # Pass through SAE if provided
            if sae is not None:
                _, latent_acts, _, _ = sae(latent_acts)
            
            # Write directly to memmap
            batch_size_actual = latent_acts.shape[0]
            activations_mmap[current_idx:current_idx+batch_size_actual] = latent_acts.cpu().numpy()
            current_idx += batch_size_actual
            
            # Explicit cleanup
            del latent_acts, patch_batch
            activations_dict.clear()
            if device == 'cuda':
                torch.cuda.empty_cache()
    
    hook_handle.remove()
    
    # Flush to disk
    activations_mmap.flush()
    del activations_mmap
    gc.collect()
    
    # Return metadata instead of loading all data
    return {
        'path': mmap_file,
        'shape': (num_patches, num_neurons),
        'dtype': 'float32'
    }


def get_top_k_patches_per_neuron(mmap_info, patches, k=100, 
                                save_dir=None, experiment_name="default"):
    """
    For each neuron, find the top-k most activating patches.
    Memory-efficient version that processes one neuron at a time.
    """
    if save_dir is None:
        save_dir = Path(f"/scratch/inf0/user/dbagci/pure/{experiment_name}")
    else:
        save_dir = Path(save_dir)
    
    save_dir.mkdir(parents=True, exist_ok=True)
    top_k_file = save_dir / f"{experiment_name}_top_{k}_patches_per_neuron.pkl"
    
    # Load if exists
    if top_k_file.exists():
        print(f"Loading cached top-k patches from {top_k_file}")
        with open(top_k_file, 'rb') as f:
            return pickle.load(f)
    
    # Open memmap in read-only mode
    num_patches, num_neurons = mmap_info['shape']
    activations_mmap = np.memmap(mmap_info['path'], dtype=mmap_info['dtype'], 
                                  mode='r', shape=mmap_info['shape'])
    
    print(f"Finding top-{k} patches per neuron...")
    top_k_patches_per_neuron = {}
    
    # Process one neuron at a time
    for neuron_idx in tqdm(range(num_neurons), desc="Processing neurons"):
        neuron_activations = torch.from_numpy(activations_mmap[:, neuron_idx].copy())
        
        # Get top-k indices
        top_k_indices = torch.topk(neuron_activations, k=min(k, num_patches)).indices
        
        # Collect corresponding patches
        top_patches = [patches[idx] for idx in top_k_indices.numpy()]
        top_k_patches_per_neuron[neuron_idx] = top_patches
        
        del neuron_activations
    
    # Clean up memmap
    del activations_mmap
    gc.collect()
    
    # Save
    print(f"Saving top-k patches to {top_k_file}")
    with open(top_k_file, 'wb') as f:
        pickle.dump(top_k_patches_per_neuron, f)
    
    return top_k_patches_per_neuron


def compute_all_clip_embeddings_batched(top_k_patches_dict, device='cuda', batch_size=64, 
                                        save_dir=None, experiment_name="default"):
    """
    Compute CLIP embeddings for all neurons' patches in batched fashion.
    Saves embeddings per neuron immediately to avoid memory issues.
    """
    if save_dir is None:
        save_dir = Path(f"/scratch/inf0/user/dbagci/pure/{experiment_name}")
    else:
        save_dir = Path(save_dir)
    
    save_dir.mkdir(parents=True, exist_ok=True)
    embeddings_file = save_dir / f"{experiment_name}_clip_embeddings.pkl"
    
    # Load if exists
    if embeddings_file.exists():
        print(f"Loading cached CLIP embeddings from {embeddings_file}")
        with open(embeddings_file, 'rb') as f:
            return pickle.load(f)
    
    print("Computing CLIP embeddings (batched)...")
    
    # Load CLIP model
    clip_model, preprocess = clip.load("ViT-B/16", device=device)
    clip_model.eval()
    
    embeddings_per_neuron = {}
    
    # Process each neuron separately
    for neuron_idx in tqdm(sorted(top_k_patches_dict.keys()), desc="Computing embeddings per neuron"):
        patches = top_k_patches_dict[neuron_idx]
        neuron_embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(patches), batch_size):
                batch = patches[i:i+batch_size]
                batch_tensors = torch.stack([preprocess(p) for p in batch]).to(device)
                embeddings = clip_model.encode_image(batch_tensors)
                embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
                neuron_embeddings.append(embeddings.cpu())
                
                del batch_tensors, embeddings
                if device == 'cuda':
                    torch.cuda.empty_cache()
        
        embeddings_per_neuron[neuron_idx] = torch.cat(neuron_embeddings, dim=0)
        del neuron_embeddings
        gc.collect()
    
    # Save embeddings
    print(f"Saving CLIP embeddings to {embeddings_file}")
    with open(embeddings_file, 'wb') as f:
        pickle.dump(embeddings_per_neuron, f)
    
    # Cleanup
    del clip_model
    if device == 'cuda':
        torch.cuda.empty_cache()
    gc.collect()
    
    return embeddings_per_neuron

from hdbscan import HDBSCAN

def compute_clustering_metrics_cosine(embeddings, k=3):
    """
    Cluster embeddings using HDBSCAN with cosine similarity.
    """
    # Normalize embeddings
    embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
    embeddings_np = embeddings.numpy()
    num_samples = embeddings_np.shape[0]
    
    # if num_samples < min_cluster_size:
    #     return {
    #         'intra_cluster_dist': None,
    #         'inter_cluster_dist': None,
    #         'num_clusters': 0,
    #         'cluster_sizes': [],
    #         'valid': False
    #     }
    
    clusterer = HDBSCAN()
    cluster_labels = clusterer.fit_predict(embeddings_np)
    
    # Filter out noise points (label -1)
    valid_mask = cluster_labels != -1
    valid_labels = cluster_labels[valid_mask]
    valid_embeddings = embeddings_np[valid_mask]
    
    if len(valid_labels) == 0:
        return {
            'intra_cluster_dist': None,
            'inter_cluster_dist': None,
            'num_clusters': 0,
            'cluster_sizes': [],
            'valid': False
        }
    
    unique_clusters, cluster_counts = np.unique(valid_labels, return_counts=True)
    num_clusters = len(unique_clusters)
    
    if num_clusters < 2:
        return {
            'intra_cluster_dist': None,
            'inter_cluster_dist': None,
            'num_clusters': num_clusters,
            'cluster_sizes': cluster_counts.tolist(),
            'valid': False
        }
    
    # Compute cluster centers (mean of normalized vectors)
    cluster_centers = []
    for cluster_id in unique_clusters:
        cluster_mask = valid_labels == cluster_id
        cluster_points = valid_embeddings[cluster_mask]
        center = np.mean(cluster_points, axis=0)
        # Normalize center
        center = center / np.linalg.norm(center)
        cluster_centers.append(center)
    cluster_centers = np.array(cluster_centers)
    
    # Compute cosine distances
    distances = cosine_distances(valid_embeddings)
    
    # Intra-cluster distances
    intra_cluster_dists = []
    for cluster_id in unique_clusters:
        cluster_mask = valid_labels == cluster_id
        cluster_indices = np.where(cluster_mask)[0]
        if len(cluster_indices) > 1:
            cluster_dists = distances[np.ix_(cluster_indices, cluster_indices)]
            triu_indices = np.triu_indices_from(cluster_dists, k=1)
            intra_dists = cluster_dists[triu_indices]
            intra_cluster_dists.extend(intra_dists)
    
    avg_intra_cluster_dist = np.mean(intra_cluster_dists) if intra_cluster_dists else None
    
    # Inter-cluster distances
    if len(cluster_centers) > 1:
        centroid_distances = cosine_distances(cluster_centers)
        triu_indices = np.triu_indices_from(centroid_distances, k=1)
        inter_cluster_dists = centroid_distances[triu_indices]
        avg_inter_cluster_dist = np.mean(inter_cluster_dists)
    else:
        avg_inter_cluster_dist = None
    
    return {
        'intra_cluster_dist': avg_intra_cluster_dist,
        'inter_cluster_dist': avg_inter_cluster_dist,
        'num_clusters': num_clusters,
        'cluster_sizes': cluster_counts.tolist(),
        'valid': True
    }


def compute_metrics_sequential(embeddings_per_neuron, k=3, min_cluster_size=5):
    """
    Sequential processing of clustering metrics to avoid memory issues.
    """
    per_neuron_results = {}
    
    for neuron_idx, embeddings in tqdm(embeddings_per_neuron.items(), desc="Clustering neurons"):
        metrics = compute_clustering_metrics_cosine(
            embeddings, 
            k=k
        )
        per_neuron_results[neuron_idx] = metrics
        
        del embeddings
        gc.collect()
    
    return per_neuron_results


def compute_neuron_clustering_metric(
    model,
    target_layer,
    val_loader,
    k=100,
    patch_size=64,
    experiment_name="default",
    save_dir=None,
    device='cuda',
    sae=None,
    min_cluster_size=5,
    min_samples=3,
    recompute_patches=False,
    recompute_activations=False,
    recompute_embeddings=False,
    return_top_patches=False,
    n_workers=4
):
    """
    Main function to compute the clustering metric for all neurons.
    
    Returns:
        avg_intra: Average intra-cluster distance
        avg_inter: Average inter-cluster distance
        sep_ratio: Separation ratio (inter/intra)
        top_k_patches: (Optional) Dictionary of top patches per neuron
    """
    if save_dir is None:
        save_dir = Path(f"/scratch/inf0/user/dbagci/pure/{experiment_name}")
    else:
        save_dir = Path(save_dir)
    
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Extract patches
    patch_cache_dir = Path("/scratch/inf0/user/dbagci/pure")
    if recompute_patches:
        patch_file = patch_cache_dir / f"imagenet_patches_size{patch_size}.pkl"
        if patch_file.exists():
            patch_file.unlink()
    
    patches, patch_metadata = extract_patches_from_imagenet(
        val_loader, 
        patch_size=patch_size, 
        save_dir=patch_cache_dir,
        device=device
    )
    
    # Step 2: Extract activations
    activation_file = save_dir / f"{experiment_name}_patch_activations.pt"
    
    if recompute_activations or not activation_file.exists():
        print("Computing patch activations...")
        activations = extract_patch_activations(
            model, patches, patch_metadata, target_layer,
            device=device, sae=sae, experiment_name=experiment_name
        )
        torch.save(activations, activation_file)
    else:
        print(f"Loading cached activations from {activation_file}")
        activations = torch.load(activation_file)
    
    # Step 3: Get top-k patches
    top_k_patches = get_top_k_patches_per_neuron(
        activations, patches, k=k, 
        save_dir=save_dir, 
        experiment_name=experiment_name
    )
    
    # Cleanup memmap
    if 'path' in activations and Path(activations['path']).exists():
        Path(activations['path']).unlink()
        print(f"Cleaned up memmap file")
    
    # Step 4: Compute CLIP embeddings
    if recompute_embeddings:
        embeddings_file = save_dir / f"{experiment_name}_clip_embeddings.pkl"
        if embeddings_file.exists():
            embeddings_file.unlink()
    
    embeddings_per_neuron = compute_all_clip_embeddings_batched(
        top_k_patches, 
        device=device, 
        batch_size=64,
        save_dir=save_dir,
        experiment_name=experiment_name
    )
    
    # Step 5: Compute clustering metrics
    print("Computing clustering metrics (sequential)...")
    per_neuron_results = compute_metrics_sequential(
        embeddings_per_neuron,
        k=3,
        min_cluster_size=min_cluster_size
    )
    
    del embeddings_per_neuron
    gc.collect()
    
    # Step 6: Aggregate results
    valid_intra_dists = []
    valid_inter_dists = []
    
    for neuron_idx, metrics in per_neuron_results.items():
        if metrics['valid'] and metrics['intra_cluster_dist'] is not None:
            valid_intra_dists.append(metrics['intra_cluster_dist'])
        if metrics['valid'] and metrics['inter_cluster_dist'] is not None:
            valid_inter_dists.append(metrics['inter_cluster_dist'])
    
    avg_intra = np.mean(valid_intra_dists) if valid_intra_dists else None
    avg_inter = np.mean(valid_inter_dists) if valid_inter_dists else None
    
    if avg_intra is not None and avg_inter is not None:
        sep_ratio = avg_inter / avg_intra
    else:
        sep_ratio = None
    
    # Save results
    results_file = save_dir / f"{experiment_name}_clustering_metrics.json"
    with open(results_file, 'w') as f:
        json_results = {
            'avg_intra_cluster_dist': float(avg_intra) if avg_intra else None,
            'avg_inter_cluster_dist': float(avg_inter) if avg_inter else None,
            'separation_ratio': float(sep_ratio) if sep_ratio else None,
            'num_valid_neurons': len(valid_intra_dists),
            'total_neurons': len(per_neuron_results),
        }
        json.dump(json_results, f, indent=2)
    
    print(f"\nResults saved to {results_file}")
    print(f"Average Intra-cluster Distance: {avg_intra:.6f}" if avg_intra else "N/A")
    print(f"Average Inter-cluster Distance: {avg_inter:.6f}" if avg_inter else "N/A")
    print(f"Separation Ratio: {sep_ratio:.6f}" if sep_ratio else "N/A")
    
    if return_top_patches:
        return avg_intra, avg_inter, sep_ratio, top_k_patches
    else:
        return avg_intra, avg_inter, sep_ratio