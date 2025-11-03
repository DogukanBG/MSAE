#!/bin/sh
#SBATCH -p gpu17
#SBATCH -t 16:00:00
#SBATCH -o /BS/disentanglement/work/Disentanglement/jobs/sae_training/dino_sae-%j.out
#SBATCH --gres gpu:1

export WANDB_API_KEY="a6f51864caf2d973f427aef4ccc067fd64c91bcd"

DATA_DIR="/scratch/inf0/user/mparcham/ILSVRC2012/val_categorized"
DATASET="imagenet"
MODEL="dinov2_vitl14" #"dinov2_vitl14"
SAVE_DIR="/BS/disentanglement/work"
BATCH_SIZE=16384

DT="/BS/disentanglement/work/sae/imagenet_dinov2_vitl14_embeddings_train.npy"
DS="/BS/disentanglement/work/sae/imagenet_dinov2_vitl14_embeddings_val.npy"
EPOCHS=30

#uv run /BS/disentanglement/work/Disentanglement/MSAE/precompute_activations.py --model "$MODEL" --dataset "$DATASET" --data-path "$DATA_DIR" --save-dir "$SAVE_DIR" --batch-size $BATCH_SIZE --split val
# M="ReLUSAE" # "TopKSAE", "BatchTopKSAE"
# ACTIVATION='ReLU_003' # 'TopKReLU_64'
# EXPANSION_FACTOR=8
# uv run /BS/disentanglement/work/Disentanglement/MSAE/train.py -dt "$DT" -ds "$DS" -m "$M" -a "$ACTIVATION" -ef "$EXPANSION_FACTOR" -e "$EPOCHS" -bs $BATCH_SIZE
# #uv run /BS/disentanglement/work/Disentanglement/MSAE/train.py -dt "$DT" -ds "$DS" -m "$M" -a "$ACTIVATION" -ef "$EXPANSION_FACTOR" -e "$EPOCHS"

# EXPANSION_FACTOR=16
# uv run /BS/disentanglement/work/Disentanglement/MSAE/train.py -dt "$DT" -ds "$DS" -m "$M" -a "$ACTIVATION" -ef "$EXPANSION_FACTOR" -e "$EPOCHS" -bs $BATCH_SIZE
# #uv run /BS/disentanglement/work/Disentanglement/MSAE/train.py -dt "$DT" -ds "$DS" -m "$M" -a "$ACTIVATION" -ef "$EXPANSION_FACTOR" -e "$EPOCHS"

# EXPANSION_FACTOR=32
# uv run /BS/disentanglement/work/Disentanglement/MSAE/train.py -dt "$DT" -ds "$DS" -m "$M" -a "$ACTIVATION" -ef "$EXPANSION_FACTOR" -e "$EPOCHS" -bs $BATCH_SIZE
# #uv run /BS/disentanglement/work/Disentanglement/MSAE/train.py -dt "$DT" -ds "$DS" -m "$M" -a "$ACTIVATION" -ef "$EXPANSION_FACTOR" -e "$EPOCHS"


# M="TopKSAE" # "TopKSAE", "BatchTopKSAE"
# ACTIVATION='TopKReLU_64' # 'TopKReLU_64'
# EXPANSION_FACTOR=8
# uv run /BS/disentanglement/work/Disentanglement/MSAE/train.py -dt "$DT" -ds "$DS" -m "$M" -a "$ACTIVATION" -ef "$EXPANSION_FACTOR" -e "$EPOCHS" -bs $BATCH_SIZE
# #uv run /BS/disentanglement/work/Disentanglement/MSAE/train.py -dt "$DT" -ds "$DS" -m "$M" -a "$ACTIVATION" -ef "$EXPANSION_FACTOR" -e "$EPOCHS"

# EXPANSION_FACTOR=16
# uv run /BS/disentanglement/work/Disentanglement/MSAE/train.py -dt "$DT" -ds "$DS" -m "$M" -a "$ACTIVATION" -ef "$EXPANSION_FACTOR" -e "$EPOCHS" -bs $BATCH_SIZE
# #uv run /BS/disentanglement/work/Disentanglement/MSAE/train.py -dt "$DT" -ds "$DS" -m "$M" -a "$ACTIVATION" -ef "$EXPANSION_FACTOR" -e "$EPOCHS"

# EXPANSION_FACTOR=32
# uv run /BS/disentanglement/work/Disentanglement/MSAE/train.py -dt "$DT" -ds "$DS" -m "$M" -a "$ACTIVATION" -ef "$EXPANSION_FACTOR" -e "$EPOCHS" -bs $BATCH_SIZE
# #uv run /BS/disentanglement/work/Disentanglement/MSAE/train.py -dt "$DT" -ds "$DS" -m "$M" -a "$ACTIVATION" -ef "$EXPANSION_FACTOR" -e "$EPOCHS"

# M="TopKSAE" # "TopKSAE", "BatchTopKSAE"
# ACTIVATION='TopKReLU_256' # 'TopKReLU_64'
# EXPANSION_FACTOR=8
# uv run /BS/disentanglement/work/Disentanglement/MSAE/train.py -dt "$DT" -ds "$DS" -m "$M" -a "$ACTIVATION" -ef "$EXPANSION_FACTOR" -e "$EPOCHS" -bs $BATCH_SIZE
# #uv run /BS/disentanglement/work/Disentanglement/MSAE/train.py -dt "$DT" -ds "$DS" -m "$M" -a "$ACTIVATION" -ef "$EXPANSION_FACTOR" -e "$EPOCHS"

# EXPANSION_FACTOR=16
# uv run /BS/disentanglement/work/Disentanglement/MSAE/train.py -dt "$DT" -ds "$DS" -m "$M" -a "$ACTIVATION" -ef "$EXPANSION_FACTOR" -e "$EPOCHS" -bs $BATCH_SIZE
# #uv run /BS/disentanglement/work/Disentanglement/MSAE/train.py -dt "$DT" -ds "$DS" -m "$M" -a "$ACTIVATION" -ef "$EXPANSION_FACTOR" -e "$EPOCHS"

# EXPANSION_FACTOR=32
# uv run /BS/disentanglement/work/Disentanglement/MSAE/train.py -dt "$DT" -ds "$DS" -m "$M" -a "$ACTIVATION" -ef "$EXPANSION_FACTOR" -e "$EPOCHS" -bs $BATCH_SIZE
# #uv run /BS/disentanglement/work/Disentanglement/MSAE/train.py -dt "$DT" -ds "$DS" -m "$M" -a "$ACTIVATION" -ef "$EXPANSION_FACTOR" -e "$EPOCHS"

# M="MSAE_UW" # "TopKSAE", "BatchTopKSAE"
# ACTIVATION='' # 'TopKReLU_64'
# EXPANSION_FACTOR=8
# uv run /BS/disentanglement/work/Disentanglement/MSAE/train.py -dt "$DT" -ds "$DS" -m "$M" -a "$ACTIVATION" -ef "$EXPANSION_FACTOR" -e "$EPOCHS" -bs $BATCH_SIZE
# #uv run /BS/disentanglement/work/Disentanglement/MSAE/train.py -dt "$DT" -ds "$DS" -m "$M" -a "$ACTIVATION" -ef "$EXPANSION_FACTOR" -e "$EPOCHS"

# EXPANSION_FACTOR=16
# uv run /BS/disentanglement/work/Disentanglement/MSAE/train.py -dt "$DT" -ds "$DS" -m "$M" -a "$ACTIVATION" -ef "$EXPANSION_FACTOR" -e "$EPOCHS" -bs $BATCH_SIZE
# #uv run /BS/disentanglement/work/Disentanglement/MSAE/train.py -dt "$DT" -ds "$DS" -m "$M" -a "$ACTIVATION" -ef "$EXPANSION_FACTOR" -e "$EPOCHS"

# EXPANSION_FACTOR=32
# uv run /BS/disentanglement/work/Disentanglement/MSAE/train.py -dt "$DT" -ds "$DS" -m "$M" -a "$ACTIVATION" -ef "$EXPANSION_FACTOR" -e "$EPOCHS" -bs $BATCH_SIZE
# #uv run /BS/disentanglement/work/Disentanglement/MSAE/train.py -dt "$DT" -ds "$DS" -m "$M" -a "$ACTIVATION" -ef "$EXPANSION_FACTOR" -e "$EPOCHS"

# M="MSAE_RW" # "TopKSAE", "BatchTopKSAE"
# ACTIVATION='' # 'TopKReLU_64'
# EXPANSION_FACTOR=8
# uv run /BS/disentanglement/work/Disentanglement/MSAE/train.py -dt "$DT" -ds "$DS" -m "$M" -a "$ACTIVATION" -ef "$EXPANSION_FACTOR" -e "$EPOCHS" -bs $BATCH_SIZE
# #uv run /BS/disentanglement/work/Disentanglement/MSAE/train.py -dt "$DT" -ds "$DS" -m "$M" -a "$ACTIVATION" -ef "$EXPANSION_FACTOR" -e "$EPOCHS"

# EXPANSION_FACTOR=16
# uv run /BS/disentanglement/work/Disentanglement/MSAE/train.py -dt "$DT" -ds "$DS" -m "$M" -a "$ACTIVATION" -ef "$EXPANSION_FACTOR" -e "$EPOCHS" -bs $BATCH_SIZE
# #uv run /BS/disentanglement/work/Disentanglement/MSAE/train.py -dt "$DT" -ds "$DS" -m "$M" -a "$ACTIVATION" -ef "$EXPANSION_FACTOR" -e "$EPOCHS"

# EXPANSION_FACTOR=32
# uv run /BS/disentanglement/work/Disentanglement/MSAE/train.py -dt "$DT" -ds "$DS" -m "$M" -a "$ACTIVATION" -ef "$EXPANSION_FACTOR" -e "$EPOCHS" -bs $BATCH_SIZE
# #uv run /BS/disentanglement/work/Disentanglement/MSAE/train.py -dt "$DT" -ds "$DS" -m "$M" -a "$ACTIVATION" -ef "$EXPANSION_FACTOR" -e "$EPOCHS"

M="BatchTopKSAE" # "TopKSAE", "BatchTopKSAE"
ACTIVATION='TopKReLU_64' # 'TopKReLU_64'
EXPANSION_FACTOR=8
uv run /BS/disentanglement/work/Disentanglement/MSAE/train.py -dt "$DT" -ds "$DS" -m "$M" -a "$ACTIVATION" -ef "$EXPANSION_FACTOR" -e "$EPOCHS" -bs $BATCH_SIZE
#uv run /BS/disentanglement/work/Disentanglement/MSAE/train.py -dt "$DT" -ds "$DS" -m "$M" -a "$ACTIVATION" -ef "$EXPANSION_FACTOR" -e "$EPOCHS"

EXPANSION_FACTOR=16
uv run /BS/disentanglement/work/Disentanglement/MSAE/train.py -dt "$DT" -ds "$DS" -m "$M" -a "$ACTIVATION" -ef "$EXPANSION_FACTOR" -e "$EPOCHS" -bs $BATCH_SIZE
#uv run /BS/disentanglement/work/Disentanglement/MSAE/train.py -dt "$DT" -ds "$DS" -m "$M" -a "$ACTIVATION" -ef "$EXPANSION_FACTOR" -e "$EPOCHS"

EXPANSION_FACTOR=32
uv run /BS/disentanglement/work/Disentanglement/MSAE/train.py -dt "$DT" -ds "$DS" -m "$M" -a "$ACTIVATION" -ef "$EXPANSION_FACTOR" -e "$EPOCHS" -bs $BATCH_SIZE
#uv run /BS/disentanglement/work/Disentanglement/MSAE/train.py -dt "$DT" -ds "$DS" -m "$M" -a "$ACTIVATION" -ef "$EXPANSION_FACTOR" -e "$EPOCHS"

M="BatchTopKSAE" # "TopKSAE", "BatchTopKSAE"
ACTIVATION='TopKReLU_256' # 'TopKReLU_64'
EXPANSION_FACTOR=8
uv run /BS/disentanglement/work/Disentanglement/MSAE/train.py -dt "$DT" -ds "$DS" -m "$M" -a "$ACTIVATION" -ef "$EXPANSION_FACTOR" -e "$EPOCHS" -bs $BATCH_SIZE
#uv run /BS/disentanglement/work/Disentanglement/MSAE/train.py -dt "$DT" -ds "$DS" -m "$M" -a "$ACTIVATION" -ef "$EXPANSION_FACTOR" -e "$EPOCHS"

EXPANSION_FACTOR=16
uv run /BS/disentanglement/work/Disentanglement/MSAE/train.py -dt "$DT" -ds "$DS" -m "$M" -a "$ACTIVATION" -ef "$EXPANSION_FACTOR" -e "$EPOCHS" -bs $BATCH_SIZE
#uv run /BS/disentanglement/work/Disentanglement/MSAE/train.py -dt "$DT" -ds "$DS" -m "$M" -a "$ACTIVATION" -ef "$EXPANSION_FACTOR" -e "$EPOCHS"

EXPANSION_FACTOR=32
uv run /BS/disentanglement/work/Disentanglement/MSAE/train.py -dt "$DT" -ds "$DS" -m "$M" -a "$ACTIVATION" -ef "$EXPANSION_FACTOR" -e "$EPOCHS" -bs $BATCH_SIZE
#uv run /BS/disentanglement/work/Disentanglement/MSAE/train.py -dt "$DT" -ds "$DS" -m "$M" -a "$ACTIVATION" -ef "$EXPANSION_FACTOR" -e "$EPOCHS"
