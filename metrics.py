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
            
    if topk < 2:
        raise ValueError("CKNNA requires topk >= 2")
    
    if topk is None:
        topk = feats_A.shape[0] - 1
                        
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

import gc
def monosemanticity_score(experiment_name, model, sae, target_layer, val_loader, device='cuda'):
    """
    Compute monosemanticity score for neural network features.
    
    Args:
        experiment_name: Name for saving/loading cached files
        model: Neural network model to analyze
        sae: Sparse autoencoder (optional, can be None)
        target_layer: Layer name to hook for activation extraction
        val_loader: DataLoader for validation images
        feature_extractor: Name of feature extractor (e.g., "dinov2_b")
        device: Device to run computations on
    
    Returns:
        Average monosemanticity score across all channels
    """
    
    # Create directory for cached files
    cache_dir = Path(f"/BS/disentanglement/work/ms_score/")
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    embeddings_path = cache_dir / f"embeddings_dinov2.pt"
    activations_path = cache_dir / f"{experiment_name}_activations_{target_layer}.pt"
    
    # Extract or load DINOv2 embeddings (for similarity computation)
    if embeddings_path.exists():
        print("Loading cached DINOv2 embeddings...")
        embeddings = torch.load(embeddings_path)
    else:
        print("Computing DINOv2 embeddings...")
        embeddings = compute_dinov2_embeddings(val_loader, device)
        torch.save(embeddings, embeddings_path)
        print(f"Saved embeddings to {embeddings_path}")
    
    # Extract or load activations from target layer
    if activations_path.exists():
        print("Loading cached activations...")
        activations = torch.load(activations_path)
    else:
        print("Extracting latent activations...")
        activations = extract_latents(model, sae, target_layer, val_loader, device)
        torch.save(activations, activations_path)
        print(f"Saved activations to {activations_path}")
    
    torch.cuda.empty_cache()
    gc.collect()
    
    print("Computing monosemanticity scores...")
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Activations shape: {activations.shape}")
    
    num_images, embed_dim = embeddings.shape
    num_neurons = activations.shape[1]
    
    # Normalize embeddings for cosine similarity
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    
    # Scale activations to 0-1 per neuron
    min_values = activations.min(dim=0, keepdim=True)[0]
    max_values = activations.max(dim=0, keepdim=True)[0]
    activations = (activations - min_values) / (max_values - min_values + 1e-8)
    
    # Move to CPU for computation
    # embeddings = embeddings.cpu()
    # activations = activations.cpu()
    
    # # Initialize accumulators
    # weighted_cosine_similarity_sum = torch.zeros(num_neurons)
    # weight_sum = torch.zeros(num_neurons)
    # batch_size = 50000  # Process pairs in batches
    
    # for i in tqdm(range(num_images), desc="Processing image pairs"):
    #     for j_start in range(i + 1, num_images, batch_size):
    #         j_end = min(j_start + batch_size, num_images)
            
    #         embeddings_i = embeddings[i].to(device)  # (embed_dim)
    #         embeddings_j = embeddings[j_start:j_end].to(device)  # (batch_size, embed_dim)
    #         activations_i = activations[i].to(device)  # (num_neurons)
    #         activations_j = activations[j_start:j_end].to(device)  # (batch_size, num_neurons)
            
    #         # Compute cosine similarity between embeddings
    #         cosine_similarities = F.cosine_similarity(
    #             embeddings_i.unsqueeze(0).expand(j_end - j_start, -1),
    #             embeddings_j,
    #             dim=1
    #         )  # (batch_size,)
            
    #         # Compute weights from activations
    #         weights = activations_i.unsqueeze(0) * activations_j  # (batch_size, num_neurons)
            
    #         # Weight similarities by activation products
    #         weighted_cosine_similarities = weights * cosine_similarities.unsqueeze(1)  # (batch_size, num_neurons)
            
    #         # Accumulate
    #         weighted_cosine_similarity_sum += weighted_cosine_similarities.sum(dim=0).cpu()
    #         weight_sum += weights.sum(dim=0).cpu()
    
    # # Compute final monosemanticity scores
    # monosemanticity = torch.where(weight_sum != 0, weighted_cosine_similarity_sum / weight_sum, torch.nan)
    embeddings = embeddings.to(device)
    activations = activations.to(device)

    # Normalize embeddings once
    embeddings_norm = F.normalize(embeddings, p=2, dim=1)  # (num_images, embed_dim)

    # Initialize accumulators
    weighted_cosine_similarity_sum = torch.zeros(num_neurons, device=device)
    weight_sum = torch.zeros(num_neurons, device=device)

    chunk_size = 100  # Adjust based on GPU memory

    for i_start in tqdm(range(0, num_images, chunk_size), desc="Processing chunks"):
        i_end = min(i_start + chunk_size, num_images)
        
        # Get chunk of embeddings and activations
        emb_chunk = embeddings_norm[i_start:i_end]  # (chunk_size, embed_dim)
        act_chunk = activations[i_start:i_end]  # (chunk_size, num_neurons)
        
        # Only compute with images that come AFTER this chunk (j > i)
        for j_start in range(i_end, num_images, chunk_size):
            j_end = min(j_start + chunk_size, num_images)
            
            emb_j = embeddings_norm[j_start:j_end]  # (chunk_j, embed_dim)
            act_j = activations[j_start:j_end]  # (chunk_j, num_neurons)
            
            # Compute cosine similarities for all pairs in chunks
            cosine_sims = emb_chunk @ emb_j.T  # (chunk_i, chunk_j)
            
            # Compute activation products
            # act_chunk: (chunk_i, num_neurons) -> (chunk_i, 1, num_neurons)
            # act_j: (chunk_j, num_neurons) -> (1, chunk_j, num_neurons)
            act_products = act_chunk.unsqueeze(1) * act_j.unsqueeze(0)  # (chunk_i, chunk_j, num_neurons)
            
            # Weight similarities
            weighted_sims = act_products * cosine_sims.unsqueeze(2)  # (chunk_i, chunk_j, num_neurons)
            
            # Accumulate
            weighted_cosine_similarity_sum += weighted_sims.sum(dim=(0, 1))
            weight_sum += act_products.sum(dim=(0, 1))
        
        # Handle pairs within the same chunk (i < j within chunk)
        if i_end - i_start > 1:
            cosine_sims = emb_chunk @ emb_chunk.T  # (chunk_size, chunk_size)
            act_products = act_chunk.unsqueeze(1) * act_chunk.unsqueeze(0)  # (chunk_size, chunk_size, num_neurons)
            
            # Create upper triangle mask (exclude diagonal)
            mask = torch.triu(torch.ones(i_end - i_start, i_end - i_start, device=device), diagonal=1)
            mask = mask.unsqueeze(2)  # (chunk_size, chunk_size, 1)
            
            weighted_sims = act_products * cosine_sims.unsqueeze(2) * mask
            
            weighted_cosine_similarity_sum += weighted_sims.sum(dim=(0, 1))
            weight_sum += (act_products * mask).sum(dim=(0, 1))

    # Compute final scores
    monosemanticity = torch.where(weight_sum != 0, weighted_cosine_similarity_sum / weight_sum, torch.nan)
                    #weighted_cosine_similarity_sum / (num_images * (num_images - 1) / 2) 
    monosemanticity = monosemanticity.cpu()

    monosemanticity_with_dead_neurons = weighted_cosine_similarity_sum / (num_images * (num_images - 1) / 2) 
    
    # Save results
    output_dir = Path(f"/BS/disentanglement/work/Disentanglement/MSAE/ms_scores/")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    # torch.save(monosemanticity, output_dir / "all_neurons_scores.pth")
    
    # Compute statistics
    is_nan = torch.isnan(monosemanticity)
    nan_count = is_nan.sum()
    monosemanticity_mean = torch.mean(monosemanticity[~is_nan])
    monosemanticity_mean_with_dead_neurons_mean = torch.mean(monosemanticity_with_dead_neurons[~is_nan])
    monosemanticity_std = torch.std(monosemanticity[~is_nan])
    
    print(f"\nMonosemanticity: {monosemanticity_mean.item():.6f} +- {monosemanticity_std.item():.6f}")
    print(f"\nMonosemanticity with Dead Neurons: {monosemanticity_mean_with_dead_neurons_mean.item():.6f}")
    print(f"Dead neurons: {nan_count.item()}")
    print(f"Total neurons: {num_neurons}")
    
    # Save results
    output_dir = Path(f"/BS/disentanglement/work/Disentanglement/MSAE/ms_scores/")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    # torch.save(monosemanticity, output_dir / "all_neurons_scores.pth")
    
    
    # Filter out NaNs for ranking
    valid_indices = ~torch.isnan(monosemanticity)
    valid_monosemanticity = monosemanticity[valid_indices]
    valid_indices = torch.nonzero(valid_indices).squeeze()
    
    # Get top 10 highest and lowest monosemantic neurons
    top_10_values, top_10_indices = torch.topk(valid_monosemanticity, min(10, len(valid_monosemanticity)))
    bottom_10_values, bottom_10_indices = torch.topk(valid_monosemanticity, min(10, len(valid_monosemanticity)), largest=False)
    
    # Map indices back to original positions
    top_10_indices = valid_indices[top_10_indices]
    bottom_10_indices = valid_indices[bottom_10_indices]
    
    # Print results
    print("\nTop 10 most monosemantic neurons:")
    for i, (idx, val) in enumerate(zip(top_10_indices, top_10_values)):
         print(f"{i + 1}. Neuron {idx.item()} - {val.item():.6f}")
    
    print("\nBottom 10 least monosemantic neurons:")
    for i, (idx, val) in enumerate(zip(bottom_10_indices, bottom_10_values)):
        print(f"{i + 1}. Neuron {idx.item()} - {val.item():.6f}")
    
    # # Save to file
    output_path = output_dir / f"{experiment_name}_metric_stats.txt"
    with open(output_path, "w") as file:
        file.write(f"Monosemanticity: {monosemanticity_mean.item():.6f} +- {monosemanticity_std.item():.6f}\n")
        file.write(f"Monosemanticity with Dead Neurons: {monosemanticity_mean_with_dead_neurons_mean.item():.6f}\n")
        file.write(f"Dead neurons: {nan_count.item()}\n")
        file.write(f"Total neurons: {num_neurons}\n\n")
        
        file.write("Top 10 most monosemantic neurons:\n")
        for idx, val in zip(top_10_indices, top_10_values):
            file.write(f"Neuron {idx.item()} - {val.item():.6f}\n")
        
        file.write("\nBottom 10 least monosemantic neurons:\n")
        for idx, val in zip(bottom_10_indices, bottom_10_values):
            file.write(f"Neuron {idx.item()} - {val.item():.6f}\n")
    
    # print(f"\nResults saved to {output_path}")
    
    return monosemanticity_mean.item()


def compute_dinov2_embeddings(val_loader, device='cuda'):
    """
    Compute DINOv2 embeddings for all images.
    Returns normalized embeddings ready for cosine similarity.
    """
    # Load DINOv2 model
    dinov2_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
    dinov2_model = dinov2_model.to(device)
    dinov2_model.eval()
    
    all_embeddings = []
    
    with torch.no_grad():
        for images, _ in tqdm(val_loader, desc="Extracting DINOv2 embeddings"):
            images = images.to(device)
            embeddings = dinov2_model(images)  # Shape: (batch_size, embed_dim)
            all_embeddings.append(embeddings.cpu())
    
    # Concatenate all embeddings
    all_embeddings = torch.cat(all_embeddings, dim=0)  # Shape: (num_images, embed_dim)
    
    # Clean up
    dinov2_model = dinov2_model.to("cpu")
    del dinov2_model
    torch.cuda.empty_cache()
    
    return all_embeddings


def extract_latents(model, sae, target_layer, val_loader, device='cuda'):
    """
    Extract latent activations from target layer, optionally passing through SAE.
    Returns raw activations (scaling done later).
    """
    activations = {}
    
    def hook_fn(module, input, output):
        activations["output"] = output.detach()
    
    # Register hook on target layer
    layer = get_layer_by_path(model, target_layer)
    hook_handle = layer.register_forward_hook(hook_fn)
    
    model = model.to(device)
    model.eval()
    
    if sae is not None:
        sae = sae.to(device)
        sae.eval()
    
    all_latents = []
    
    with torch.no_grad():
        for images, _ in tqdm(val_loader, desc="Extracting latents"):
            images = images.to(device)
            
            # Forward pass to capture activations
            _ = model(images)
            latent_acts = activations["output"]
            
            # Handle different activation shapes
            if latent_acts.dim() == 4:  # Conv layer: (B, C, H, W)
                latent_acts = latent_acts.sum(dim=(2, 3))
            elif latent_acts.dim() == 3:  # Transformer: (B, T, C)
                latent_acts = latent_acts[:, 0, :]  # Take CLS token
            
            # Pass through SAE if provided
            if sae is not None:
                sparse_outputs, latent_acts, outputs, representations = sae(latent_acts)
            
            all_latents.append(latent_acts.cpu())
    
    # Remove hook
    hook_handle.remove()
    
    # Stack all latents
    all_latents = torch.cat(all_latents, dim=0)  # Shape: (num_images, num_channels)
    
    return all_latents
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pickle
import hdbscan
from sklearn.metrics import pairwise_distances
import clip
from PIL import Image
import json


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
    
    Args:
        val_loader: DataLoader for ImageNet validation set
        patch_size: Size of square patches to extract
        save_dir: Directory to save/load patches from
        device: Device to use for processing
    
    Returns:
        patches: List of PIL Images (patches)
        metadata: Dict with patch metadata (original image idx, position)
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
                    
                    # Convert to PIL Image for saving/loading
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
                              batch_size=256, device='cuda', sae=None):
    """
    Extract activations for all patches through the target layer.
    Returns path to memmap file instead of loading all activations.
    
    Returns:
        mmap_info: Dict with {'path': mmap_file, 'shape': (num_patches, num_neurons), 'dtype': 'float32'}
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
            
            # Free GPU memory
            del latent_acts
            if device == 'cuda':
                torch.cuda.empty_cache()
    
    hook_handle.remove()
    
    # Flush to disk
    activations_mmap.flush()
    del activations_mmap
    
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
    
    Args:
        mmap_info: Dict with memmap file info from extract_patch_activations
        patches: List of PIL Image patches
        k: Number of top patches to keep per neuron
        save_dir: Directory to save top-k patches
        experiment_name: Name of experiment for organizing files
    
    Returns:
        top_k_patches_per_neuron: Dict mapping neuron_idx -> list of PIL Images
    """
    if save_dir is None:
        save_dir = Path(f"/scratch/inf0/user/dbagci/pure/{experiment_name}")
    else:
        save_dir = Path(f"/scratch/inf0/user/dbagci/pure/{experiment_name}")
    
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
    
    # Process one neuron at a time - only loads one column into memory
    for neuron_idx in tqdm(range(num_neurons), desc="Processing neurons"):
        # Load activations for this neuron only (shape: num_patches)
        neuron_activations = torch.from_numpy(activations_mmap[:, neuron_idx].copy())
        
        # Get top-k indices
        top_k_indices = torch.topk(neuron_activations, k=min(k, num_patches)).indices
        
        # Collect corresponding patches
        top_patches = [patches[idx] for idx in top_k_indices.numpy()]
        top_k_patches_per_neuron[neuron_idx] = top_patches
        
        # Free memory
        del neuron_activations
    
    # Clean up memmap
    del activations_mmap
    
    # Save
    print(f"Saving top-k patches to {top_k_file}")
    with open(top_k_file, 'wb') as f:
        pickle.dump(top_k_patches_per_neuron, f)
    
    return top_k_patches_per_neuron


def compute_clip_embeddings(patches, device='cuda', batch_size=100):
    """
    Compute CLIP embeddings for a list of patches.
    
    Args:
        patches: List of PIL Images
        device: Device to use
        batch_size: Batch size for CLIP encoding
    
    Returns:
        embeddings: Tensor of shape (num_patches, clip_dim)
    """
    # Load CLIP model
    clip_model, preprocess = clip.load("ViT-B/16", device=device)
    clip_model.eval()
    
    all_embeddings = []
    
    with torch.no_grad():
        for i in range(0, len(patches), batch_size):
            batch_patches = patches[i:i+batch_size]
            
            # Preprocess patches
            batch_tensors = torch.stack([preprocess(patch) for patch in batch_patches])
            batch_tensors = batch_tensors.to(device)
            
            # Encode with CLIP
            embeddings = clip_model.encode_image(batch_tensors)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)  # Normalize
            
            all_embeddings.append(embeddings.cpu())
    
    return torch.cat(all_embeddings, dim=0)


def compute_clustering_metrics(embeddings, min_cluster_size=5, min_samples=3):
    """
    Cluster embeddings using K-Means and compute intra/inter-cluster distances.
    
    Args:
        embeddings: Tensor of shape (num_samples, embedding_dim)
        k: Number of clusters for K-Means (default: 3)
    
    Returns:
        metrics: Dict with intra_cluster_dist, inter_cluster_dist, num_clusters, cluster_sizes
    """
    # Normalize embeddings to unit length
    k = 3
    embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
    embeddings_np = embeddings.numpy()
    
    num_samples = embeddings_np.shape[0]
    
    # Check if we have enough samples
    print("Embeddings Shape", embeddings_np.shape)
    if num_samples < k:
        return {
            'intra_cluster_dist': None,
            'inter_cluster_dist': None,
            'num_clusters': 0,
            'cluster_sizes': [],
            'valid': False
        }
    
    # Perform K-Means clustering
    from sklearn.cluster import KMeans
    
    kmeans = KMeans(n_clusters=k, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings_np)
    cluster_centers = kmeans.cluster_centers_
    
    # Count samples in each cluster
    unique_clusters, cluster_counts = np.unique(cluster_labels, return_counts=True)
    num_clusters = len(unique_clusters)
    
    if num_clusters < 2:
        # This shouldn't happen with K-Means, but check anyway
        return {
            'intra_cluster_dist': None,
            'inter_cluster_dist': None,
            'num_clusters': num_clusters,
            'cluster_sizes': cluster_counts.tolist(),
            'valid': False
        }
    
    # Compute pairwise distances
    distances = pairwise_distances(embeddings_np, metric='euclidean')
    
    # Compute intra-cluster distances (average distance within each cluster)
    intra_cluster_dists = []
    for cluster_id in unique_clusters:
        cluster_mask = cluster_labels == cluster_id
        cluster_indices = np.where(cluster_mask)[0]
        
        if len(cluster_indices) > 1:
            # Get distances within this cluster
            cluster_dists = distances[np.ix_(cluster_indices, cluster_indices)]
            # Take upper triangle (excluding diagonal)
            triu_indices = np.triu_indices_from(cluster_dists, k=1)
            intra_dists = cluster_dists[triu_indices]
            intra_cluster_dists.extend(intra_dists)
    
    avg_intra_cluster_dist = np.mean(intra_cluster_dists) if intra_cluster_dists else None
    
    # Compute inter-cluster distances (average distance between cluster centroids)
    if len(cluster_centers) > 1:
        centroid_distances = pairwise_distances(cluster_centers, metric='euclidean')
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
    recompute_activations=False
):
    """
    Main function to compute the clustering metric for all neurons in a layer.
    
    Args:
        model: Neural network model
        target_layer: Layer path to analyze
        val_loader: DataLoader for ImageNet validation set
        k: Number of top activating patches per neuron
        patch_size: Size of patches to extract
        experiment_name: Name for organizing cached files
        save_dir: Directory to save results
        device: Device to use
        sae: Optional SAE to pass activations through
        min_cluster_size: HDBSCAN parameter
        min_samples: HDBSCAN parameter
        recompute_patches: Force recompute patches
        recompute_activations: Force recompute activations
    
    Returns:
        results: Dict with average metrics and per-neuron results
    """
    if save_dir is None:
        save_dir = Path(f"/scratch/inf0/user/dbagci/pure/{experiment_name}")
    else:
        save_dir = Path(save_dir)
    
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Extract patches from ImageNet (model-independent)
    patch_cache_dir = Path("/scratch/inf0/user/dbagci/pure") #save_dir / "patches"
    if recompute_patches:
        import shutil
        if patch_cache_dir.exists():
            shutil.rmtree(patch_cache_dir)
    
    patches, patch_metadata = extract_patches_from_imagenet(
        val_loader, 
        patch_size=patch_size, 
        save_dir=patch_cache_dir,
        device=device
    )
    
    # Step 2: Extract activations for all patches (model-dependent)
    activation_file = save_dir / f"{experiment_name}_patch_activations_{target_layer.replace('.', '_')}.pt"
    
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
    
    # Step 3: Get top-k patches per neuron
    top_k_patches = get_top_k_patches_per_neuron(
        activations, patches, k=k, 
        save_dir=save_dir, 
        experiment_name=experiment_name
    )
    
    # Step 4: For each neuron, compute CLIP embeddings and clustering metrics
    print("Computing clustering metrics per neuron...")
    num_neurons = len(top_k_patches)
    
    # Setup checkpointing
    checkpoint_file = save_dir / f"{experiment_name}_neuron_analysis_checkpoint.pkl"
    checkpoint_interval = 100  # Save every 100 neurons
    
    # Load existing checkpoint if available
    if checkpoint_file.exists():
        print(f"Loading checkpoint from {checkpoint_file}")
        with open(checkpoint_file, 'rb') as f:
            checkpoint = pickle.load(f)
        per_neuron_results = checkpoint['per_neuron_results']
        start_neuron = checkpoint['last_completed_neuron'] + 1
        print(f"Resuming from neuron {start_neuron}/{num_neurons}")
    else:
        per_neuron_results = {}
        start_neuron = 0
    
    valid_intra_dists = []
    valid_inter_dists = []
    
    for neuron_idx in tqdm(range(start_neuron, num_neurons), desc="Analyzing neurons", initial=start_neuron, total=num_neurons):
        neuron_patches = top_k_patches[neuron_idx]
        
        # Compute CLIP embeddings
        embeddings = compute_clip_embeddings(neuron_patches, device=device)
        
        # Compute clustering metrics
        metrics = compute_clustering_metrics(
            embeddings, 
            min_cluster_size=min_cluster_size,
            min_samples=min_samples
        )
        
        per_neuron_results[neuron_idx] = metrics
        
        # Save checkpoint periodically
        if (neuron_idx + 1) % checkpoint_interval == 0:
            checkpoint = {
                'per_neuron_results': per_neuron_results,
                'last_completed_neuron': neuron_idx,
                'num_neurons': num_neurons
            }
            with open(checkpoint_file, 'wb') as f:
                pickle.dump(checkpoint, f)
            print(f"\nCheckpoint saved at neuron {neuron_idx + 1}/{num_neurons}")
    
    # Collect valid metrics for averaging (check all results, not just new ones)
    for neuron_idx in range(num_neurons):
        if neuron_idx in per_neuron_results:
            metrics = per_neuron_results[neuron_idx]
            if metrics['valid'] and metrics['intra_cluster_dist'] is not None:
                valid_intra_dists.append(metrics['intra_cluster_dist'])
            if metrics['valid'] and metrics['inter_cluster_dist'] is not None:
                valid_inter_dists.append(metrics['inter_cluster_dist'])
    
    # Clean up checkpoint file after completion
    if checkpoint_file.exists():
        checkpoint_file.unlink()
        print("Checkpoint file removed after successful completion")
    
    # Step 5: Compute averages
    avg_intra_cluster_dist = np.mean(valid_intra_dists) if valid_intra_dists else None
    avg_inter_cluster_dist = np.mean(valid_inter_dists) if valid_inter_dists else None
    
    # Compute ratio (higher is better - more separation between clusters)
    if avg_intra_cluster_dist is not None and avg_inter_cluster_dist is not None:
        separation_ratio = avg_inter_cluster_dist / avg_intra_cluster_dist
    else:
        separation_ratio = None
    
    results = {
        'avg_intra_cluster_dist': avg_intra_cluster_dist,
        'avg_inter_cluster_dist': avg_inter_cluster_dist,
        'separation_ratio': separation_ratio,
        'num_valid_neurons': len(valid_intra_dists),
        'total_neurons': num_neurons,
        'per_neuron_results': per_neuron_results
    }
    
    # Save results
    results_file = save_dir / f"{experiment_name}_clustering_metrics_{target_layer.replace('.', '_')}.json"
    with open(results_file, 'w') as f:
        # Convert to JSON-serializable format
        json_results = {
            'avg_intra_cluster_dist': float(avg_intra_cluster_dist) if avg_intra_cluster_dist else None,
            'avg_inter_cluster_dist': float(avg_inter_cluster_dist) if avg_inter_cluster_dist else None,
            'separation_ratio': float(separation_ratio) if separation_ratio else None,
            'num_valid_neurons': len(valid_intra_dists),
            'total_neurons': num_neurons,
        }
        json.dump(json_results, f, indent=2)
    
    print(f"\nResults saved to {results_file}")
    print(f"\nAverage Intra-cluster Distance: {avg_intra_cluster_dist:.4f}" if avg_intra_cluster_dist else "N/A")
    print(f"Average Inter-cluster Distance: {avg_inter_cluster_dist:.4f}" if avg_inter_cluster_dist else "N/A")
    print(f"Separation Ratio: {separation_ratio:.4f}" if separation_ratio else "N/A")
    print(f"Valid Neurons: {len(valid_intra_dists)}/{num_neurons}")
    
    return avg_intra_cluster_dist, avg_inter_cluster_dist, separation_ratio


def compute_neuron_clustering_metric_optimized(
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
    clip_batch_size=200,  # Increased from 100 for better throughput
    neurons_per_batch=50   # Process multiple neurons together
):
    """
    Optimized version that loads CLIP once, batches neurons, and uses mixed precision.
    """
    if save_dir is None:
        save_dir = Path(f"/scratch/inf0/user/dbagci/pure/{experiment_name}")
    else:
        save_dir = Path(save_dir)
    
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Extract patches from ImageNet (model-independent)
    patch_cache_dir = Path("/scratch/inf0/user/dbagci/pure")
    if recompute_patches:
        import shutil
        if patch_cache_dir.exists():
            shutil.rmtree(patch_cache_dir)
    
    patches, patch_metadata = extract_patches_from_imagenet(
        val_loader, 
        patch_size=patch_size, 
        save_dir=patch_cache_dir,
        device=device
    )
    
    # Step 2: Extract activations for all patches (model-dependent)
    activation_file = save_dir / f"{experiment_name}_patch_activations_{target_layer.replace('.', '_')}.pt"
    
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
    
    # Step 3: Get top-k patches per neuron
    top_k_patches = get_top_k_patches_per_neuron(
        activations, patches, k=k, 
        save_dir=save_dir, 
        experiment_name=experiment_name
    )
    
    # Step 4: OPTIMIZED - Load CLIP once and batch processing
    print("Computing clustering metrics per neuron (optimized)...")
    num_neurons = len(top_k_patches)
    per_neuron_results = {}
    
    valid_intra_dists = []
    valid_inter_dists = []
    
    # Load CLIP model once outside the loop
    import clip
    clip_model, clip_preprocess = clip.load("ViT-B/16", device=device)
    clip_model.eval()
    
    # Enable mixed precision for CLIP inference
    from torch.cuda.amp import autocast
    
    # Process neurons in batches
    neuron_indices = list(range(num_neurons))
    
    for batch_start in tqdm(range(0, num_neurons, neurons_per_batch), desc="Analyzing neuron batches"):
        batch_end = min(batch_start + neurons_per_batch, num_neurons)
        neuron_batch = neuron_indices[batch_start:batch_end]
        
        # Collect all patches for this batch of neurons
        all_batch_patches = []
        neuron_patch_ranges = []  # Track which patches belong to which neuron
        
        for neuron_idx in neuron_batch:
            neuron_patches = top_k_patches[neuron_idx]
            start_idx = len(all_batch_patches)
            all_batch_patches.extend(neuron_patches)
            end_idx = len(all_batch_patches)
            neuron_patch_ranges.append((neuron_idx, start_idx, end_idx))
        
        if not all_batch_patches:
            continue
            
        # Compute CLIP embeddings for all patches in this batch
        all_embeddings = []
        with torch.no_grad(), autocast():
            for i in range(0, len(all_batch_patches), clip_batch_size):
                batch_patches = all_batch_patches[i:i+clip_batch_size]
                
                # Preprocess patches
                batch_tensors = torch.stack([clip_preprocess(patch) for patch in batch_patches])
                batch_tensors = batch_tensors.to(device)
                
                # Encode with CLIP using mixed precision
                embeddings = clip_model.encode_image(batch_tensors)
                embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)  # Normalize
                
                all_embeddings.append(embeddings.cpu())
        
        # Concatenate all embeddings for this neuron batch
        if all_embeddings:
            combined_embeddings = torch.cat(all_embeddings, dim=0)
            
            # Process each neuron's embeddings individually for clustering
            for neuron_idx, start_idx, end_idx in neuron_patch_ranges:
                neuron_embeddings = combined_embeddings[start_idx:end_idx]
                
                # Compute clustering metrics
                metrics = compute_clustering_metrics(
                    neuron_embeddings, 
                    min_cluster_size=min_cluster_size,
                    min_samples=min_samples
                )
                
                per_neuron_results[neuron_idx] = metrics
                
                # Collect valid metrics for averaging
                if metrics['valid'] and metrics['intra_cluster_dist'] is not None:
                    valid_intra_dists.append(metrics['intra_cluster_dist'])
                if metrics['valid'] and metrics['inter_cluster_dist'] is not None:
                    valid_inter_dists.append(metrics['inter_cluster_dist'])
    
    # Step 5: Compute averages
    avg_intra_cluster_dist = np.mean(valid_intra_dists) if valid_intra_dists else None
    avg_inter_cluster_dist = np.mean(valid_inter_dists) if valid_inter_dists else None
    
    # Compute ratio (higher is better - more separation between clusters)
    if avg_intra_cluster_dist is not None and avg_inter_cluster_dist is not None:
        separation_ratio = avg_inter_cluster_dist / avg_intra_cluster_dist
    else:
        separation_ratio = None
    
    results = {
        'avg_intra_cluster_dist': avg_intra_cluster_dist,
        'avg_inter_cluster_dist': avg_inter_cluster_dist,
        'separation_ratio': separation_ratio,
        'num_valid_neurons': len(valid_intra_dists),
        'total_neurons': num_neurons,
        'per_neuron_results': per_neuron_results
    }
    
    # Save results
    results_file = save_dir / f"{experiment_name}_clustering_metrics_{target_layer.replace('.', '_')}_optimized.json"
    with open(results_file, 'w') as f:
        json_results = {
            'avg_intra_cluster_dist': float(avg_intra_cluster_dist) if avg_intra_cluster_dist else None,
            'avg_inter_cluster_dist': float(avg_inter_cluster_dist) if avg_inter_cluster_dist else None,
            'separation_ratio': float(separation_ratio) if separation_ratio else None,
            'num_valid_neurons': len(valid_intra_dists),
            'total_neurons': num_neurons,
        }
        json.dump(json_results, f, indent=2)
    
    print(f"\nResults saved to {results_file}")
    print(f"\nAverage Intra-cluster Distance: {avg_intra_cluster_dist:.4f}" if avg_intra_cluster_dist else "N/A")
    print(f"Average Inter-cluster Distance: {avg_inter_cluster_dist:.4f}" if avg_inter_cluster_dist else "N/A")
    print(f"Separation Ratio: {separation_ratio:.4f}" if separation_ratio else "N/A")
    print(f"Valid Neurons: {len(valid_intra_dists)}/{num_neurons}")
    
    return avg_intra_cluster_dist, avg_inter_cluster_dist, separation_ratio
