#!/bin/bash
#SBATCH -p gpu22
#SBATCH --job-name=linear_probe
#SBATCH -o /BS/disentanglement/work/Disentanglement/jobs/sae_training/linear_eval-%j.out
#SBATCH --time=8:00:00
#SBATCH --gres gpu:1

STORE_DIR="/BS/disentanglement/work/sae/extract_embeddings"


## DINOv2
TRAIN_DATA="/BS/disentanglement/work/sae/linear_probe/imagenet_dinov2_vitl14_embeddings_train.npy"
DATA="/BS/disentanglement/work/sae/linear_probe/imagenet_dinov2_vitl14_embeddings_val.npy"
TRAIN_LABELS="/BS/disentanglement/work/sae/linear_probe/imagenet_dinov2_vitl14_labels_train.npy"
DATA_LABELS="/BS/disentanglement/work/sae/linear_probe/imagenet_dinov2_vitl14_labels_val.npy"

TAU=0.03
MODEL="/BS/disentanglement/work/sae/8192_1024_ReLU_003_False_False_0.0_imagenet_dinov2_vitl14_embeddings.pth"
#uv run /BS/disentanglement/work/Disentanglement/MSAE/linear_eval.py -m $MODEL -d $TRAIN_DATA -o $DATA_LABELS -t $TRAIN_LABELS -e $DATA --eval_hc True --tau $TAU

TAU=0.02
MODEL="/BS/disentanglement/work/sae/16384_1024_ReLU_003_False_False_0.0_imagenet_dinov2_vitl14_embeddings.pth"
#uv run /BS/disentanglement/work/Disentanglement/MSAE/linear_eval.py -m $MODEL -d $TRAIN_DATA -o $DATA_LABELS -t $TRAIN_LABELS -e $DATA --eval_hc True --tau $TAU

echo "DINOv2 evaluated"

##ViTB16
TRAIN_DATA="/BS/disentanglement/work/sae/linear_probe/imagenet_vit_b_16_embeddings_train.npy"
DATA="/BS/disentanglement/work/sae/linear_probe/imagenet_vit_b_16_embeddings_val.npy"
TRAIN_LABELS="/BS/disentanglement/work/sae/linear_probe/imagenet_vit_b_16_labels_train.npy"
DATA_LABELS="/BS/disentanglement/work/sae/linear_probe/imagenet_vit_b_16_labels_val.npy"

TAU=0.03
MODEL="/BS/disentanglement/work/sae/6144_768_ReLU_003_False_False_0.0_imagenet_vit_b_16_embeddings_train.pth"
#uv run /BS/disentanglement/work/Disentanglement/MSAE/linear_eval.py -m $MODEL -d $TRAIN_DATA -o $DATA_LABELS -t $TRAIN_LABELS -e $DATA --eval-hc True --tau $TAU

TAU=0.02
MODEL="/BS/disentanglement/work/sae/12288_768_ReLU_003_False_False_0.0_imagenet_vit_b_16_embeddings_train.pth"
#uv run /BS/disentanglement/work/Disentanglement/MSAE/linear_eval.py -m $MODEL -d $TRAIN_DATA -o $DATA_LABELS -t $TRAIN_LABELS -e $DATA --eval_hc True --tau $TAU

echo "ViT-b/16 evaluated"

##ResNet50
TRAIN_DATA="/BS/disentanglement/work/sae/linear_probe/imagenet_resnet50_embeddings_train.npy"
DATA="/BS/disentanglement/work/sae/linear_probe/imagenet_resnet50_embeddings_val.npy"
TRAIN_LABELS="/BS/disentanglement/work/sae/linear_probe/imagenet_resnet50_labels_train.npy"
DATA_LABELS="/BS/disentanglement/work/sae/linear_probe/imagenet_resnet50_labels_val.npy"

TAU=0.03
MODEL="/BS/disentanglement/work/sae/16384_2048_ReLU_003_False_False_0.0_imagenet_resnet50_embeddings.pth"
#uv run /BS/disentanglement/work/Disentanglement/MSAE/linear_eval.py -m $MODEL -d $TRAIN_DATA -o $DATA_LABELS -t $TRAIN_LABELS -e $DATA --eval-hc True --tau $TAU

TAU=0.02
MODEL="/BS/disentanglement/work/sae/32768_2048_ReLU_003_False_False_0.0_imagenet_resnet50_embeddings.pth"
uv run /BS/disentanglement/work/Disentanglement/MSAE/linear_eval.py -m $MODEL -d $TRAIN_DATA -o $DATA_LABELS -t $TRAIN_LABELS -e $DATA --eval_hc True --tau $TAU

echo "ResNet50 evaluated"