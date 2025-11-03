#!/bin/bash
#SBATCH -p gpu22
#SBATCH --job-name=fms_hc
#SBATCH -o /BS/disentanglement/work/Disentanglement/jobs/sae_training/fms-%j.out
#SBATCH --time=1:00:00
#SBATCH --gres gpu:1

STORE_DIR="/BS/disentanglement/work/sae/extract_embeddings"


## DINOv2
MODEL="dinov2_vitl14"
TAU=0.03
SAE="/BS/disentanglement/work/sae/8192_1024_ReLU_003_False_False_0.0_imagenet_dinov2_vitl14_embeddings.pth"
uv run /BS/disentanglement/work/Disentanglement/MSAE/fms.py -m $MODEL --sae $SAE  

TAU=0.02
SAE="/BS/disentanglement/work/sae/16384_1024_ReLU_003_False_False_0.0_imagenet_dinov2_vitl14_embeddings.pth"
#uv run /BS/disentanglement/work/Disentanglement/MSAE/fms.py -m $MODEL --sae $SAE  

#echo "DINOv2 evaluated"

##ViTB16
MODEL="vit_b_16"
TAU=0.03
SAE="/BS/disentanglement/work/sae/6144_768_ReLU_003_False_False_0.0_imagenet_vit_b_16_embeddings_train.pth"
#uv run /BS/disentanglement/work/Disentanglement/MSAE/fms.py -m $MODEL --eval-hc True --tau $TAU

TAU=0.02
SAE="/BS/disentanglement/work/sae/12288_768_ReLU_003_False_False_0.0_imagenet_vit_b_16_embeddings_train.pth"
#uv run /BS/disentanglement/work/Disentanglement/MSAE/fms.py -m $MODEL --eval-hc True --tau $TAU

TAU=0.01
SAE="/BS/disentanglement/work/sae/12288_768_ReLU_003_False_False_0.0_imagenet_vit_b_16_embeddings_train.pth"
#uv run /BS/disentanglement/work/Disentanglement/MSAE/fms.py -m $MODEL --eval-hc True --tau $TAU

echo "ViT-b/16 HC evaluated"

##ResNet50
MODEL="resnet50"
TAU=0.03
SAE="/BS/disentanglement/work/sae/16384_2048_ReLU_003_False_False_0.0_imagenet_resnet50_embeddings.pth"
#uv run /BS/disentanglement/work/Disentanglement/MSAE/fms.py -m $MODEL --eval-hc True --tau $TAU

TAU=0.02
SAE="/BS/disentanglement/work/sae/32768_2048_ReLU_003_False_False_0.0_imagenet_resnet50_embeddings.pth"
#uv run /BS/disentanglement/work/Disentanglement/MSAE/fms.py -m $MODEL --eval-hc True --tau $TAU


echo "ResNet50 HC evaluated"