import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import silhouette_score
from hdbscan import HDBSCAN
import clip
import pickle
import json

def get_layer_by_path(model, layer_path):
    """Navigate to a layer using dot notation path."""
    layer = model
    for attr in layer_path.split('.'):
        layer = getattr(layer, attr)
    return layer

def extract_patches_from_images(images, patch_size=64, stride=None):
    """
    Extract non-overlapping or overlapping patches from images.
    
    Args:
        images: Tensor of shape (B, C, H, W)
        patch_size: Size of square patches
        stride: Stride for patch extraction (defaults to patch_size for non-overlapping)
    
    Returns:
        patches: Tensor of shape (num_patches, C, patch_size, patch_size)
        patch_info: List of dicts with image_idx, row, col for each patch
    """
    if stride is None:
        stride = patch_size
    
    B, C, H, W = images.shape
    patches = []
    patch_info = []
    
    for img_idx in range(B):
        img = images[img_idx]
        
        # Calculate number of patches
        n_rows = (H - patch_size) // stride + 1
        n_cols = (W - patch_size) // stride + 1
        
        for row in range(n_rows):
            for col in range(n_cols):
                top = row * stride
                left = col * stride
                
                patch = img[:, top:top+patch_size, left:left+patch_size]
                patches.append(patch)
                patch_info.append({
                    'image_idx': img_idx,
                    'row': row,
                    'col': col,
                    'top': top,
                    'left': left
                })
    
    patches = torch.stack(patches)
    return patches, patch_info

def extract_top_k_patches(model, sae, target_layer, val_loader, k=100, 
                         patch_size=64, stride=None, save_dir=None, 
                         device='cuda', use_cache=True):
    """
    Extract top-k highest activating patches for each neuron.
    Cuts images into patches first, then processes them through the model.
    
    Args:
        model: The neural network model
        sae: Sparse autoencoder (optional)
        target_layer: String path to target layer
        val_loader: DataLoader for validation set
        k: Number of top patches to extract per neuron
        patch_size: Size of patches to extract
        stride: Stride for patch extraction (None = non-overlapping)
        save_dir: Directory to save patches
        device: Device to run on
        use_cache: Whether to load cached patches if available
    
    Returns:
        top_patches_dict: Dict mapping neuron_idx -> list of top patch info
        patch_data: Dict with actual patch tensors for CLIP encoding
    """
    save_dir = Path(save_dir) if save_dir else None
    
    # Check for cached data
    if use_cache and save_dir is not None:
        cache_file = save_dir / "patch_activations_cache.pkl"
        if cache_file.exists():
            print(f"Loading cached patch data from {cache_file}")
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            
            # Load patch images
            patch_data = {}
            for neuron_idx in tqdm(cache_data['top_patches_dict'].keys(), 
                                  desc="Loading cached patches"):
                neuron_dir = save_dir / f"neuron_{neuron_idx:04d}"
                if neuron_dir.exists():
                    patches = []
                    for i in range(min(k, len(list(neuron_dir.glob("patch_*.png"))))):
                        patch_path = list(neuron_dir.glob(f"patch_{i:03d}_*.png"))[0]
                        patch_img = Image.open(patch_path)
                        patch_tensor = transforms.ToTensor()(patch_img)
                        patches.append(patch_tensor)
                    if patches:
                        patch_data[neuron_idx] = torch.stack(patches)
            
            print(f"Loaded cached data for {len(patch_data)} neurons")
            return cache_data['top_patches_dict'], patch_data
    
    activations = {}
    
    def hook_fn(module, input, output):
        activations["output"] = output.detach()
    
    # Register hook
    layer = get_layer_by_path(model, target_layer)
    hook_handle = layer.register_forward_hook(hook_fn)
    
    model = model.to(device)
    model.eval()
    if sae is not None:
        sae = sae.to(device)
        sae.eval()
    
    # Store all patch activations per neuron
    neuron_activations = {}
    all_patches_list = []
    all_patch_info_list = []
    
    print("Extracting patches and computing activations...")
    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(tqdm(val_loader, desc="Processing batches")):
            images = images.to(device)
            
            # Extract patches from images
            patches, patch_info = extract_patches_from_images(
                images, patch_size=patch_size, stride=stride
            )
            
            # Store patches and info
            all_patches_list.append(patches.cpu())
            
            # Add batch offset to patch info
            offset = len(all_patch_info_list)
            for info in patch_info:
                info['global_patch_idx'] = offset + len(all_patch_info_list) - offset
                info['batch_idx'] = batch_idx
            all_patch_info_list.extend(patch_info)
            
            # Resize patches to model's expected input size
            # Assuming model expects 224x224 (adjust if needed)
            patches_resized = F.interpolate(
                patches.to(device), 
                size=(224, 224), 
                mode='bilinear', 
                align_corners=False
            )
            
            # Process patches in sub-batches to avoid OOM
            sub_batch_size = 32
            num_patches = patches_resized.shape[0]
            
            for sub_start in range(0, num_patches, sub_batch_size):
                sub_end = min(sub_start + sub_batch_size, num_patches)
                sub_patches = patches_resized[sub_start:sub_end]
                
                # Forward pass
                _ = model(sub_patches)
                latent_acts = activations["output"]
                
                # Handle different activation shapes
                if latent_acts.dim() == 4:  # Conv layer: (B, C, H, W)
                    latent_acts = latent_acts.amax(dim=(2, 3))
                elif latent_acts.dim() == 3:  # Transformer: (B, T, C)
                    latent_acts = latent_acts[:, 0, :]  # Take CLS token
                
                # Pass through SAE if provided
                if sae is not None:
                    sparse_outputs, latent_acts, outputs, representations = sae(latent_acts)
                
                # Store activations per neuron
                num_neurons = latent_acts.shape[1]
                for neuron_idx in range(num_neurons):
                    if neuron_idx not in neuron_activations:
                        neuron_activations[neuron_idx] = []
                    
                    # Store activation for each patch
                    for local_patch_idx in range(latent_acts.shape[0]):
                        global_patch_idx = offset + sub_start + local_patch_idx
                        act_value = latent_acts[local_patch_idx, neuron_idx].item()
                        
                        neuron_activations[neuron_idx].append({
                            'global_patch_idx': global_patch_idx,
                            'activation': act_value,
                            'patch_info': all_patch_info_list[global_patch_idx]
                        })
    
    hook_handle.remove()
    
    # Concatenate all patches
    all_patches = torch.cat(all_patches_list, dim=0)
    
    # Extract top-k patches per neuron
    print(f"\nExtracting top-{k} patches per neuron...")
    top_patches_dict = {}
    patch_data = {}
    
    num_neurons = len(neuron_activations)
    
    for neuron_idx in tqdm(range(num_neurons), desc="Saving top patches"):
        # Sort by activation value
        sorted_acts = sorted(neuron_activations[neuron_idx], 
                           key=lambda x: x['activation'], 
                           reverse=True)
        
        # Take top-k
        top_k_acts = sorted_acts[:k]
        top_patches_dict[neuron_idx] = top_k_acts
        
        # Extract patches
        patches = []
        for act_info in top_k_acts:
            patch = all_patches[act_info['global_patch_idx']]
            patches.append(patch)
        
        patch_data[neuron_idx] = torch.stack(patches)
        
        # Save patches if directory provided
        if save_dir is not None:
            neuron_dir = save_dir / f"neuron_{neuron_idx:04d}"
            neuron_dir.mkdir(parents=True, exist_ok=True)
            
            for i, patch in enumerate(patches):
                # Convert to PIL and save
                patch_np = patch.permute(1, 2, 0).numpy()
                patch_np = (patch_np * 255).clip(0, 255).astype(np.uint8)
                img_pil = Image.fromarray(patch_np)
                img_pil.save(
                    neuron_dir / f"patch_{i:03d}_act_{top_k_acts[i]['activation']:.4f}.png"
                )
    
    # Save cache
    if save_dir is not None:
        cache_file = save_dir / "patch_activations_cache.pkl"
        print(f"\nSaving cache to {cache_file}")
        with open(cache_file, 'wb') as f:
            pickle.dump({
                'top_patches_dict': top_patches_dict,
                'num_neurons': num_neurons,
                'patch_size': patch_size,
                'k': k
            }, f)
        
        # Save metadata
        metadata = {
            'num_neurons': num_neurons,
            'patch_size': patch_size,
            'k': k,
            'stride': stride,
            'total_patches_processed': len(all_patches)
        }
        with open(save_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
    
    return top_patches_dict, patch_data

def compute_clustering_metrics_hdbscan(patch_data, min_cluster_size=5, 
                                       min_samples=3, device='cuda'):
    """
    Compute inter-cluster and intra-cluster distances using CLIP embeddings and HDBSCAN.
    
    Args:
        patch_data: Dict mapping neuron_idx -> tensor of patches (k, 3, H, W)
        min_cluster_size: Minimum cluster size for HDBSCAN
        min_samples: Minimum samples parameter for HDBSCAN
        device: Device for CLIP encoding
    
    Returns:
        metrics: Dict with clustering metrics
    """
    # Load CLIP model
    print("Loading CLIP model...")
    clip_model, preprocess = clip.load("ViT-B/16", device=device)
    
    inter_cluster_dists = []
    intra_cluster_dists = []
    silhouette_scores = []
    num_clusters_list = []
    noise_ratios = []
    
    print("\nComputing HDBSCAN clustering metrics per neuron...")
    for neuron_idx, patches in tqdm(patch_data.items(), desc="Processing neurons"):
        if len(patches) < min_cluster_size:
            print(f"Warning: Neuron {neuron_idx} has fewer patches than min_cluster_size, skipping...")
            continue
        
        # Encode patches with CLIP
        embeddings = []
        with torch.no_grad():
            for patch in patches:
                # Normalize patch to [0, 1] if needed
                if patch.max() > 1.0:
                    patch = patch / 255.0
                
                # Resize and normalize for CLIP
                patch_resized = F.interpolate(
                    patch.unsqueeze(0), 
                    size=(224, 224), 
                    mode='bilinear', 
                    align_corners=False
                ).squeeze(0)
                
                # CLIP preprocessing
                patch_normalized = transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711]
                )(patch_resized)
                
                # Encode
                image_features = clip_model.encode_image(
                    patch_normalized.unsqueeze(0).to(device)
                )
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                embeddings.append(image_features.cpu())
        
        embeddings = torch.cat(embeddings, dim=0).numpy()
        
        # Cluster with HDBSCAN
        clusterer = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric='euclidean'
        )
        labels = clusterer.fit_predict(embeddings)
        
        # Filter out noise points (label = -1)
        valid_mask = labels != -1
        valid_embeddings = embeddings[valid_mask]
        valid_labels = labels[valid_mask]
        
        # Track noise ratio
        noise_ratio = (labels == -1).sum() / len(labels)
        noise_ratios.append(noise_ratio)
        
        # Skip if too few valid clusters
        unique_labels = np.unique(valid_labels)
        num_clusters_list.append(len(unique_labels))
        
        if len(unique_labels) < 2 or len(valid_embeddings) < min_cluster_size:
            continue
        
        # Compute cluster centers
        centers = []
        for cluster_id in unique_labels:
            cluster_points = valid_embeddings[valid_labels == cluster_id]
            center = cluster_points.mean(axis=0)
            centers.append(center)
        centers = np.array(centers)
        
        # Compute intra-cluster distance (average distance to cluster center)
        intra_dists = []
        for cluster_id in unique_labels:
            cluster_points = valid_embeddings[valid_labels == cluster_id]
            center = centers[cluster_id]
            dists = np.linalg.norm(cluster_points - center, axis=1)
            intra_dists.extend(dists.tolist())
        
        if len(intra_dists) > 0:
            intra_cluster_dists.append(np.mean(intra_dists))
        
        # Compute inter-cluster distance (average distance between cluster centers)
        if len(centers) > 1:
            inter_dists = []
            for i in range(len(centers)):
                for j in range(i+1, len(centers)):
                    dist = np.linalg.norm(centers[i] - centers[j])
                    inter_dists.append(dist)
            inter_cluster_dists.append(np.mean(inter_dists))
        
        # Compute silhouette score (only for valid clusters)
        if len(valid_embeddings) >= 2 and len(unique_labels) > 1:
            try:
                silhouette = silhouette_score(valid_embeddings, valid_labels)
                silhouette_scores.append(silhouette)
            except:
                pass
    
    # Compute averages
    metrics = {
        'avg_intra_cluster_dist': np.mean(intra_cluster_dists) if intra_cluster_dists else 0.0,
        'avg_inter_cluster_dist': np.mean(inter_cluster_dists) if inter_cluster_dists else 0.0,
        'avg_silhouette_score': np.mean(silhouette_scores) if silhouette_scores else 0.0,
        'avg_num_clusters': np.mean(num_clusters_list) if num_clusters_list else 0.0,
        'avg_noise_ratio': np.mean(noise_ratios) if noise_ratios else 0.0,
        'num_neurons_processed': len(patch_data),
        'num_neurons_with_valid_clusters': len(silhouette_scores)
    }
    
    return metrics

def neuron_activation_clustering_metric(model, sae, target_layer, val_loader,
                                       k=100, patch_size=64, stride=None,
                                       min_cluster_size=5, min_samples=3,
                                       experiment_name="clustering_analysis",
                                       base_dir="/BS/disentanglement/work",
                                       device='cuda', use_cache=True):
    """
    Complete pipeline for neuron activation clustering metric using HDBSCAN.
    
    Args:
        model: Neural network model
        sae: Sparse autoencoder (optional, can be None)
        target_layer: String path to target layer
        val_loader: DataLoader for validation set
        k: Number of top patches per neuron
        patch_size: Size of extracted patches from original images
        stride: Stride for patch extraction (None = non-overlapping)
        min_cluster_size: Minimum cluster size for HDBSCAN
        min_samples: Minimum samples for HDBSCAN
        experiment_name: Name for saving results
        base_dir: Base directory for saving
        device: Device to run on
        use_cache: Whether to use cached patch data if available
    
    Returns:
        metrics: Dictionary with clustering metrics
    """
    # Create save directory
    save_dir = Path(base_dir) / experiment_name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Starting neuron activation clustering analysis...")
    print(f"Target layer: {target_layer}")
    print(f"Top-k patches: {k}")
    print(f"Patch size: {patch_size}x{patch_size}")
    print(f"Stride: {stride if stride else 'non-overlapping'}")
    print(f"HDBSCAN min_cluster_size: {min_cluster_size}")
    print(f"Save directory: {save_dir}")
    print(f"Use cache: {use_cache}")
    
    # Step 1 & 2: Extract and save top-k patches (or load from cache)
    top_patches_dict, patch_data = extract_top_k_patches(
        model, sae, target_layer, val_loader,
        k=k, patch_size=patch_size, stride=stride, 
        save_dir=save_dir, device=device, use_cache=use_cache
    )
    
    # Step 3 & 4: Compute clustering metrics using HDBSCAN
    metrics = compute_clustering_metrics_hdbscan(
        patch_data, min_cluster_size=min_cluster_size,
        min_samples=min_samples, device=device
    )
    
    # Save metrics
    with open(save_dir / "clustering_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Print results
    print("\n" + "="*60)
    print("HDBSCAN CLUSTERING METRICS RESULTS")
    print("="*60)
    print(f"Average Intra-cluster Distance: {metrics['avg_intra_cluster_dist']:.4f}")
    print(f"Average Inter-cluster Distance: {metrics['avg_inter_cluster_dist']:.4f}")
    print(f"Average Silhouette Score: {metrics['avg_silhouette_score']:.4f}")
    print(f"Average Number of Clusters: {metrics['avg_num_clusters']:.2f}")
    print(f"Average Noise Ratio: {metrics['avg_noise_ratio']:.2%}")
    print(f"Neurons Processed: {metrics['num_neurons_processed']}")
    print(f"Neurons with Valid Clusters: {metrics['num_neurons_with_valid_clusters']}")
    print("="*60)
    print("\nInterpretation:")
    print("- Lower intra-cluster distance = tighter, more coherent clusters")
    print("- Higher inter-cluster distance = more separated clusters")
    print("- Higher silhouette score = better defined clusters (range: -1 to 1)")
    print("- Lower noise ratio = more patches assigned to meaningful clusters")
    print("="*60)
    
    return metrics

# Example usage:
"""
# Load your model and data
model = ... # Your pretrained model
sae = ... # Your SAE (or None)
val_loader = ... # Your ImageNet validation dataloader

# Run the metric (first time - will compute and cache)
metrics = neuron_activation_clustering_metric(
    model=model,
    sae=sae,
    target_layer='layer4.2',
    val_loader=val_loader,
    k=100,
    patch_size=64,
    stride=32,  # 50% overlap, or None for non-overlapping
    min_cluster_size=5,
    min_samples=3,
    experiment_name="my_experiment",
    base_dir="/BS/disentanglement/work",
    device='cuda',
    use_cache=True
)

# Run again - will load from cache instantly
metrics = neuron_activation_clustering_metric(
    model=model,
    sae=sae,
    target_layer='layer4.2',
    val_loader=val_loader,
    experiment_name="my_experiment",  # Same name loads cache
    use_cache=True
)
"""