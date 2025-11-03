#!/bin/sh
#SBATCH -p gpu22
#SBATCH -t 3:00:00
#SBATCH -o /BS/disentanglement/work/Disentanglement/jobs/sae_training/sae-%A_%a.out
#SBATCH --gres gpu:1
#SBATCH --array=0-8

# Fixed parameters
DT="/BS/disentanglement/work/sae/imagenet_vit_b_16_embeddings_train.npy"
DS="/BS/disentanglement/work/sae/imagenet_vit_b_16_embeddings_val.npy"
EPOCHS=10

# Select configuration based on array task ID
case $SLURM_ARRAY_TASK_ID in
    0) M="ReLUSAE"; ACTIVATION="ReLU_003"; EXPANSION_FACTOR=8 ;;
    1) M="ReLUSAE"; ACTIVATION="ReLU_003"; EXPANSION_FACTOR=16 ;;
    2) M="ReLUSAE"; ACTIVATION="ReLU_003"; EXPANSION_FACTOR=32 ;;
    3) M="TopKSAE"; ACTIVATION="TopKReLU_64"; EXPANSION_FACTOR=8 ;;
    4) M="TopKSAE"; ACTIVATION="TopKReLU_64"; EXPANSION_FACTOR=16 ;;
    5) M="TopKSAE"; ACTIVATION="TopKReLU_64"; EXPANSION_FACTOR=32 ;;
    6) M="TopKSAE"; ACTIVATION="TopKReLU_256"; EXPANSION_FACTOR=8 ;;
    7) M="TopKSAE"; ACTIVATION="TopKReLU_256"; EXPANSION_FACTOR=16 ;;
    8) M="TopKSAE"; ACTIVATION="TopKReLU_256"; EXPANSION_FACTOR=32 ;;
esac

echo "Running: Model=$M, Activation=$ACTIVATION, ExpansionFactor=$EXPANSION_FACTOR"

uv run /BS/disentanglement/work/Disentanglement/MSAE/train.py \
    -dt "$DT" -ds "$DS" -m "$M" -a "$ACTIVATION" \
    -ef "$EXPANSION_FACTOR" -e "$EPOCHS"