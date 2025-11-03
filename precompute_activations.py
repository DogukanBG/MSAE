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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# -----------------------------
# Model Loader
# -----------------------------
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

# -----------------------------
# Dataset Loaders
# -----------------------------
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

class CelebAWrapper(Dataset):
    def __init__(self, root, split="train", transform=None):
        self.dataset = CelebA(root=root, split=split, download=True, transform=transform)
    
    def __getitem__(self, idx):
        img, _ = self.dataset[idx]
        return img
    
    def __len__(self):
        return len(self.dataset)

# -----------------------------
# Embedding Extraction
# -----------------------------

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

import torch.nn.functional as F
def extract_embeddings(model, dataloader, device, embedding_dim, save_path, layer_path, args):
    dataset_size = len(dataloader.dataset)
    memmap = np.memmap(save_path, dtype=np.float32, mode='w+', shape=(dataset_size, embedding_dim))
    idx = 0
    
    activations = {}
    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook
    
    layer = get_layer_by_path(model, layer_path)
    model = model.to(device)
    handle = layer.register_forward_hook(get_activation("layer"))
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting embeddings"):
            if isinstance(batch, (list, tuple)):
                imgs = batch[0] if len(batch[0].shape) == 4 else batch
            else:
                imgs = batch
            
            imgs = imgs.to(device)
            model(imgs)
            features = activations["layer"]
            if args.model.startswith("vit") or args.model.startswith("dinov2"):
                features = features[:, 0, :]
            elif args.model.startswith("resnet"):
                features = features.sum(dim=(2,3))
            if isinstance(features, torch.Tensor):
                features = features.squeeze()
                if features.ndim == 1:
                    features = features.unsqueeze(0)
            
            #if args.model.startswith("dinov2"):
            #    features = F.normalize(features, p=2, dim=1)  # L2 normalization

            memmap[idx:idx + features.shape[0]] = features.detach().cpu().numpy()
            idx += features.shape[0]
    
    memmap.flush()
    logger.info(f"Saved embeddings to {save_path}")
    return save_path


def extract_embeddings_and_labels(model, dataloader, device, embedding_dim, save_path, 
                                  label_save_path, layer_path, args, dataset=None):
    """
    Extract embeddings and save corresponding labels.
    
    Args:
        model: The neural network model
        dataloader: DataLoader for the dataset
        device: Device to run on
        embedding_dim: Dimension of embeddings
        save_path: Path to save embeddings (.npy)
        label_save_path: Path to save labels (.txt)
        layer_path: Path to the layer for hook
        args: Command line arguments
        dataset: Original dataset object (for ImageFolder class names)
    """
    dataset_size = len(dataloader.dataset)
    memmap = np.memmap(save_path, dtype=np.float32, mode='w+', shape=(dataset_size, embedding_dim))
    idx = 0
    
    # List to store labels
    all_labels = []
    
    activations = {}
    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook
    
    layer = get_layer_by_path(model, layer_path)
    model = model.to(device)
    handle = layer.register_forward_hook(get_activation("layer"))
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting embeddings and labels"):
            # Handle different batch formats
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                imgs, labels = batch
            else:
                imgs = batch
                labels = None
            
            imgs = imgs.to(device)
            model(imgs)
            features = activations["layer"]
            
            # Process features based on model type
            if args.model.startswith("vit") or args.model.startswith("dinov2"):
                features = features[:, 0, :]
            elif args.model.startswith("resnet"):
                features = features.sum(dim=(2, 3))
            
            if isinstance(features, torch.Tensor):
                features = features.squeeze()
                if features.ndim == 1:
                    features = features.unsqueeze(0)
            
            # Save embeddings
            memmap[idx:idx + features.shape[0]] = features.detach().cpu().numpy()
            
            # Process and save labels
            if labels is not None:
                if args.dataset.lower() == "imagenet":
                    # For ImageNet, labels are class indices
                    # Convert to class names if available
                    if hasattr(dataset, 'classes'):
                        label_names = [dataset.classes[label.item()] for label in labels]
                    else:
                        label_names = [str(label.item()) for label in labels]
                    all_labels.extend(label_names)
                
                elif args.dataset.lower() == "celeba":
                    # For CelebA, handle different target types
                    if args.celeba_target_type == "identity":
                        # Identity labels are single integers
                        label_names = [str(label.item()) for label in labels]
                    else:  # attributes
                        # Attributes are binary vectors
                        # You can either save all attributes or select specific ones
                        if args.celeba_attr_idx is not None:
                            # Use specific attribute
                            label_names = [str(label[args.celeba_attr_idx].item()) for label in labels]
                        else:
                            # Convert all attributes to string (space-separated)
                            label_names = [' '.join(map(str, label.tolist())) for label in labels]
                    all_labels.extend(label_names)
            
            idx += features.shape[0]
    
    handle.remove()
    memmap.flush()
    logger.info(f"Saved embeddings to {save_path}")
    
    # Save labels to text file
    if all_labels:
        with open(label_save_path, 'w') as f:
            for label in all_labels:
                f.write(f"{label}\n")
        logger.info(f"Saved {len(all_labels)} labels to {label_save_path}")
    else:
        logger.warning("No labels were extracted!")
    
    return save_path, label_save_path

# -----------------------------
# Main
# -----------------------------
def main(args):
    device = "cuda" #if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    model, layer_path, embedding_dim = get_model(args.model, device)
    
    if args.dataset.lower() == "imagenet":
        loader, dataset_size = create_data_loaders(args.data_path, batch_size=args.batch_size, num_workers=args.workers)
    else:
        raise ValueError(f"Unsupported dataset {args.dataset}")
    
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, f"sae/{args.dataset}_{args.model}_embeddings_{args.split}.npy")
    labels_save_path = os.path.join(args.save_dir, f"sae/{args.dataset}_{args.model}_labels_{args.split}.npy")
    
    
    extract_embeddings(model, loader, device, embedding_dim, save_path, layer_path=layer_path, args=args)
    #extract_embeddings_and_labels(model=model, dataloader=loader, device=device, embedding_dim=embedding_dim, save_path=save_path, label_save_path=labels_save_path, layer_path=layer_path, args=args, dataset="imagenet")

# -----------------------------
# Argparse
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name: imagenet, celeba")
    parser.add_argument("--data-path", type=str, required=True, help="Path to dataset root")
    parser.add_argument("--model", type=str, required=True, help="Model: resnet50, convnext_tiny, vit_b_16, dinov2_vitb14")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--save-dir", type=str, default="data")
    parser.add_argument("--split", type=str, default="train", help="Dataset split for CelebA")
    args = parser.parse_args()
    
    main(args)
