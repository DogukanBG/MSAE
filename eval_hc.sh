#!/bin/sh
#SBATCH -p gpu16
#SBATCH -t 01:00:00
#SBATCH -o /BS/disentanglement/work/Disentanglement/jobs/hc/hc_eval-%j.out
#SBATCH --gres gpu:1

DATA_DIR="/scratch/inf0/user/mparcham/ILSVRC2012/train"
DATASET="imagenet"
MODEL="vit_b_16" #"dinov2_vitl14"
SAVE_DIR="/BS/disentanglement/work"
BATCH_SIZE=4096

DT="/BS/disentanglement/work/sae/imagenet_vit_b_16_embeddings_train.npy"
DS="/BS/disentanglement/work/sae/imagenet_vit_b_16_embeddings_val.npy"
EPOCHS=10

M="ReLUSAE" # "TopKSAE", "BatchTopKSAE"
ACTIVATION='ReLU_003' # 'TopKReLU_64'
EXPANSION_FACTOR=8
uv run /BS/disentanglement/work/Disentanglement/MSAE/train.py -dt "$DT" -ds "$DS" -m "$M" -a "$ACTIVATION" -ef "$EXPANSION_FACTOR" -e "$EPOCHS" --eval_hc True
