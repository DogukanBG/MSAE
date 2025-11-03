from tqdm import tqdm
from sklearn import tree
from utils import get_layer_by_path
import matplotlib.pyplot as plt
import pickle
from tree_loader import get_tree_stats, get_root_node
from random import shuffle
import pandas as pd

#from utils.visualization import get_subdataset_for_classes
from torchvision.models import resnet50

import os

"""
Code for measuring the FMS Score on ImageNet-1K
https://arxiv.org/abs/2506.19382

Step 1.
Get ImageNet-1K features for two distinct classes: Dog and Car

Step 2.
Train Decision Tree on latents

Step 3.
Cut Decision Tree and retrain

Step 4. 
Compute FMS Score
"""

import os

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

def get_number_of_images_per_class(train_dir):
    imagenet_dir = train_dir
    number_of_images_per_class = []
    for subdir, dirs, files in os.walk(imagenet_dir):
        for dir in sorted(dirs):
            dir_for_class = os.path.join(imagenet_dir, dir)
            number_of_images_per_class.append(len([name for name in os.listdir(dir_for_class)]))
    return number_of_images_per_class

def get_subdataset_for_classes(dir, class_idx_list):
    indices_to_keep = []
    number_of_images_per_class = get_number_of_images_per_class(dir)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

    # the train transform is on purpose as for eval --> we don't use them for training but measuring stuff
    train_dataset = datasets.ImageFolder(dir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    for class_idx in class_idx_list:
        start_idx_for_class = sum(number_of_images_per_class[:class_idx])
        end_idx_for_class = sum(number_of_images_per_class[:class_idx+1])
        for x in list(range(start_idx_for_class, end_idx_for_class)):
            indices_to_keep.append(x)

    return torch.utils.data.Subset(train_dataset, indices_to_keep)

def IN1k_latents(model, sae, layer_path, dataloader_concept1, dataloader_concept2):
    
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    
    layer = get_layer_by_path(model, layer_path)
    handle = layer.register_forward_hook(get_activation('acts'))

    activations_concept1_sae = []
    #print("Extracting activations for concept1...")
    for img, label in dataloader_concept1:
        img = img.to("cuda:0")
        model(img)
        if not sae is None:
            #print(activation['acts'].shape)
            if activation['acts'].dim() == 4:
                _, latents, _, _ = sae(activation['acts'].sum(dim=(2,3)))#.sum(dim=(2,3))
            else:
                _, latents, _, _  = sae(activation['acts'][:, 0, :])
                
        else:
            if activation['acts'].dim() == 4:
                latents = activation['acts'].sum(dim=(2,3))
            else:
                latents = activation['acts'][:, 0, :]
        activations_concept1_sae.append(latents.detach().cpu())
    activations_concept1_sae = torch.cat(activations_concept1_sae, dim=0)

    #print("Extracting activations for concept2...")
    activations_concept2_sae = []
    for img, label in dataloader_concept2:
        img = img.to("cuda:0")
        model(img)
        if not sae is None:
            if activation['acts'].dim() == 4:
                _, latents, _, _ = sae(activation['acts'].sum(dim=(2,3)))
            else:
                 _, latents, _, _  = sae(activation['acts'][:, 0, :])
        else:
            if activation['acts'].dim() == 4:
                latents = activation['acts'].sum(dim=(2,3))
            else:
                latents = activation['acts'][:, 0, :]
        activations_concept2_sae.append(latents.detach().cpu())
    activations_concept2_sae = torch.cat(activations_concept2_sae, dim=0)

    #concept1_sum = []
    #for latent in tqdm(activations_concept1_sae):
    #    l_sum = latent.sum(dim=(2,3))
    #    concept1_sum.append(l_sum)

    #concept2_sum = []
    #for latent in tqdm(activations_concept2_sae):
    #    l_sum = latent.sum(dim=(2,3))
    #    concept2_sum.append(l_sum)

    all_latents = torch.cat([activations_concept1_sae, activations_concept2_sae], dim=0).numpy() #activations_concept1_sae + activations_concept2_sae
    labels = [0 for _ in range(len(activations_concept1_sae))] + [1 for _ in range(len(activations_concept2_sae))]
    handle.remove()

    return all_latents, labels

def IN1k_tree(experiment_name, model, sae, layer_path, dataloader_concept1, dataloader_concept2, file_path, plot=False):
    all_latents, labels = IN1k_latents(model, sae, layer_path, dataloader_concept1, dataloader_concept2)
    
    clf = tree.DecisionTreeClassifier(criterion="gini", max_depth=None)
    clf = clf.fit(all_latents, labels)
    #print("Decision Tree Depth",clf.get_depth())

    res = {"model": [], "depth": []}
    res["model"].append(experiment_name)
    res["depth"].append(clf.get_depth())

    if plot:
        plt.figure(figsize=(40, 20))
        tree.plot_tree(
            clf,
            proportion=False,
            class_names=["Class 1", "Class 2"],
            filled=True,
            max_depth=3
        )
        plt.savefig(f"./clf_{experiment_name}.png")

    s = pickle.dumps(clf)
    with open(f"{file_path}/{experiment_name}.pkl","wb") as f:
        f.write(s)

    df = get_tree_stats(clf=clf)
    df.to_csv(f"{file_path}/{experiment_name}_stats.csv")

    return

def cut_IN1k_tree(experiment_name, model, sae, layer_path, dataloader_concept1, dataloader_concept2, file_path, label_shuffle = False):
    all_latents, labels = IN1k_latents(model, sae, layer_path, dataloader_concept1, dataloader_concept2)

    if label_shuffle:
        shuffle(labels)

    res = pd.DataFrame()
    root_node = None
    root_nodes = []
    for i in range(10):
        ls_new = []
        if root_node is not None:
            for latent in all_latents:
                latent[root_node] = 0
                ls_new.append(latent)
            all_latents = ls_new
        
        clf = tree.DecisionTreeClassifier(criterion="gini", max_depth=3)
        clf = clf.fit(all_latents, labels)

        tree_stats = get_tree_stats(clf=clf)
        root_node = get_root_node(tree_model=clf)["feature_index"]

        tree_stats["num_cuts"] = len(root_nodes)
        res = pd.concat([res, tree_stats])

        root_nodes.append(root_node)

    res.to_csv(f"{file_path}/{experiment_name}_cut.csv")

    return

