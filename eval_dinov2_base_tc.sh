#!/bin/sh
#SBATCH -p gpu24
#SBATCH -t 12:00:00
#SBATCH -o /BS/disentanglement/work/Disentanglement/experiments/dinov2_base_sae_training/dino_sae-%j.out
#SBATCH --gres gpu:1

export WANDB_API_KEY="a6f51864caf2d973f427aef4ccc067fd64c91bcd"

DATA_DIR="/scratch/inf0/user/mparcham/ILSVRC2012/val_categorized"
DATASET="imagenet"
MODEL="dinov2_vitb14" 
SAVE_DIR="/BS/disentanglement/work"
BATCH_SIZE=512

DT="/BS/disentanglement/work/tc/imagenet_dinov2_vitb14_embeddings_train.npy"
DS="/BS/disentanglement/work/tc/imagenet_dinov2_vitb14_embeddings_val.npy"
EPOCHS=20

#uv run /BS/disentanglement/work/Disentanglement/MSAE/precompute_activations.py --model "$MODEL" --dataset "$DATASET" --data-path "$DATA_DIR" --save-dir "$SAVE_DIR" --batch-size $BATCH_SIZE --split val
# M="ReLUTC" # "TopKSAE", "BatchTopKSAE"
# ACTIVATION='ReLU_003' # 'TopKReLU_64'
# EXPANSION_FACTOR=1
# uv run /BS/disentanglement/work/Disentanglement/MSAE/train.py -dt "$DT" -ds "$DS" -m "$M" -a "$ACTIVATION" -ef "$EXPANSION_FACTOR" -e "$EPOCHS" -bs $BATCH_SIZE --train_tc
# #uv run /BS/disentanglement/work/Disentanglement/MSAE/train.py -dt "$DT" -ds "$DS" -m "$M" -a "$ACTIVATION" -ef "$EXPANSION_FACTOR" -e "$EPOCHS"

# EXPANSION_FACTOR=2
# uv run /BS/disentanglement/work/Disentanglement/MSAE/train.py -dt "$DT" -ds "$DS" -m "$M" -a "$ACTIVATION" -ef "$EXPANSION_FACTOR" -e "$EPOCHS" -bs $BATCH_SIZE --train_tc
# #uv run /BS/disentanglement/work/Disentanglement/MSAE/train.py -dt "$DT" -ds "$DS" -m "$M" -a "$ACTIVATION" -ef "$EXPANSION_FACTOR" -e "$EPOCHS"

# EXPANSION_FACTOR=4
# uv run /BS/disentanglement/work/Disentanglement/MSAE/train.py -dt "$DT" -ds "$DS" -m "$M" -a "$ACTIVATION" -ef "$EXPANSION_FACTOR" -e "$EPOCHS" -bs $BATCH_SIZE --train_tc
# #uv run /BS/disentanglement/work/Disentanglement/MSAE/train.py -dt "$DT" -ds "$DS" -m "$M" -a "$ACTIVATION" -ef "$EXPANSION_FACTOR" -e "$EPOCHS"

# EXPANSION_FACTOR=8
# uv run /BS/disentanglement/work/Disentanglement/MSAE/train.py -dt "$DT" -ds "$DS" -m "$M" -a "$ACTIVATION" -ef "$EXPANSION_FACTOR" -e "$EPOCHS" -bs $BATCH_SIZE --train_tc
# #uv run /BS/disentanglement/work/Disentanglement/MSAE/train.py -dt "$DT" -ds "$DS" -m "$M" -a "$ACTIVATION" -ef "$EXPANSION_FACTOR" -e "$EPOCHS"

# EXPANSION_FACTOR=16
# uv run /BS/disentanglement/work/Disentanglement/MSAE/train.py -dt "$DT" -ds "$DS" -m "$M" -a "$ACTIVATION" -ef "$EXPANSION_FACTOR" -e "$EPOCHS" -bs $BATCH_SIZE --train_tc
# #uv run /BS/disentanglement/work/Disentanglement/MSAE/train.py -dt "$DT" -ds "$DS" -m "$M" -a "$ACTIVATION" -ef "$EXPANSION_FACTOR" -e "$EPOCHS"

# EXPANSION_FACTOR=32
# uv run /BS/disentanglement/work/Disentanglement/MSAE/train.py -dt "$DT" -ds "$DS" -m "$M" -a "$ACTIVATION" -ef "$EXPANSION_FACTOR" -e "$EPOCHS" -bs $BATCH_SIZE --train_tc
# #uv run /BS/disentanglement/work/Disentanglement/MSAE/train.py -dt "$DT" -ds "$DS" -m "$M" -a "$ACTIVATION" -ef "$EXPANSION_FACTOR" -e "$EPOCHS"
M="ReLUTC" # "TopKSAE", "BatchTopKSAE"
ACTIVATION='ReLU_001' # 'TopKReLU_64'
LR=0.00001
EXPANSION_FACTOR=1
uv run /BS/disentanglement/work/Disentanglement/MSAE/train.py -dt "$DT" -ds "$DS" -m "$M" -a "$ACTIVATION" -ef "$EXPANSION_FACTOR" -e "$EPOCHS" -bs $BATCH_SIZE --train_tc --lr "$LR"
#uv run /BS/disentanglement/work/Disentanglement/MSAE/train.py -dt "$DT" -ds "$DS" -m "$M" -a "$ACTIVATION" -ef "$EXPANSION_FACTOR" -e "$EPOCHS"

EXPANSION_FACTOR=2
uv run /BS/disentanglement/work/Disentanglement/MSAE/train.py -dt "$DT" -ds "$DS" -m "$M" -a "$ACTIVATION" -ef "$EXPANSION_FACTOR" -e "$EPOCHS" -bs $BATCH_SIZE --train_tc --lr "$LR"
#uv run /BS/disentanglement/work/Disentanglement/MSAE/train.py -dt "$DT" -ds "$DS" -m "$M" -a "$ACTIVATION" -ef "$EXPANSION_FACTOR" -e "$EPOCHS"

EXPANSION_FACTOR=4
uv run /BS/disentanglement/work/Disentanglement/MSAE/train.py -dt "$DT" -ds "$DS" -m "$M" -a "$ACTIVATION" -ef "$EXPANSION_FACTOR" -e "$EPOCHS" -bs $BATCH_SIZE --train_tc --lr "$LR"
#uv run /BS/disentanglement/work/Disentanglement/MSAE/train.py -dt "$DT" -ds "$DS" -m "$M" -a "$ACTIVATION" -ef "$EXPANSION_FACTOR" -e "$EPOCHS"

EXPANSION_FACTOR=8
uv run /BS/disentanglement/work/Disentanglement/MSAE/train.py -dt "$DT" -ds "$DS" -m "$M" -a "$ACTIVATION" -ef "$EXPANSION_FACTOR" -e "$EPOCHS" -bs $BATCH_SIZE --train_tc --lr "$LR"
# #uv run /BS/disentanglement/work/Disentanglement/MSAE/train.py -dt "$DT" -ds "$DS" -m "$M" -a "$ACTIVATION" -ef "$EXPANSION_FACTOR" -e "$EPOCHS"


M="ReLUTC" # "TopKSAE", "BatchTopKSAE"
ACTIVATION='ReLU_003' # 'TopKReLU_64'
LR=0.00001
EXPANSION_FACTOR=1
uv run /BS/disentanglement/work/Disentanglement/MSAE/train.py -dt "$DT" -ds "$DS" -m "$M" -a "$ACTIVATION" -ef "$EXPANSION_FACTOR" -e "$EPOCHS" -bs $BATCH_SIZE --train_tc --lr "$LR"
#uv run /BS/disentanglement/work/Disentanglement/MSAE/train.py -dt "$DT" -ds "$DS" -m "$M" -a "$ACTIVATION" -ef "$EXPANSION_FACTOR" -e "$EPOCHS"

EXPANSION_FACTOR=2
uv run /BS/disentanglement/work/Disentanglement/MSAE/train.py -dt "$DT" -ds "$DS" -m "$M" -a "$ACTIVATION" -ef "$EXPANSION_FACTOR" -e "$EPOCHS" -bs $BATCH_SIZE --train_tc --lr "$LR"
#uv run /BS/disentanglement/work/Disentanglement/MSAE/train.py -dt "$DT" -ds "$DS" -m "$M" -a "$ACTIVATION" -ef "$EXPANSION_FACTOR" -e "$EPOCHS"

EXPANSION_FACTOR=4
uv run /BS/disentanglement/work/Disentanglement/MSAE/train.py -dt "$DT" -ds "$DS" -m "$M" -a "$ACTIVATION" -ef "$EXPANSION_FACTOR" -e "$EPOCHS" -bs $BATCH_SIZE --train_tc --lr "$LR"
#uv run /BS/disentanglement/work/Disentanglement/MSAE/train.py -dt "$DT" -ds "$DS" -m "$M" -a "$ACTIVATION" -ef "$EXPANSION_FACTOR" -e "$EPOCHS"

EXPANSION_FACTOR=8
uv run /BS/disentanglement/work/Disentanglement/MSAE/train.py -dt "$DT" -ds "$DS" -m "$M" -a "$ACTIVATION" -ef "$EXPANSION_FACTOR" -e "$EPOCHS" -bs $BATCH_SIZE --train_tc --lr "$LR"

M="ReLUTC" # "TopKSAE", "BatchTopKSAE"
ACTIVATION='ReLU_001' # 'TopKReLU_64'
LR=0.00005
EXPANSION_FACTOR=1
uv run /BS/disentanglement/work/Disentanglement/MSAE/train.py -dt "$DT" -ds "$DS" -m "$M" -a "$ACTIVATION" -ef "$EXPANSION_FACTOR" -e "$EPOCHS" -bs $BATCH_SIZE --train_tc --lr "$LR"
#uv run /BS/disentanglement/work/Disentanglement/MSAE/train.py -dt "$DT" -ds "$DS" -m "$M" -a "$ACTIVATION" -ef "$EXPANSION_FACTOR" -e "$EPOCHS"

EXPANSION_FACTOR=2
uv run /BS/disentanglement/work/Disentanglement/MSAE/train.py -dt "$DT" -ds "$DS" -m "$M" -a "$ACTIVATION" -ef "$EXPANSION_FACTOR" -e "$EPOCHS" -bs $BATCH_SIZE --train_tc --lr "$LR"
#uv run /BS/disentanglement/work/Disentanglement/MSAE/train.py -dt "$DT" -ds "$DS" -m "$M" -a "$ACTIVATION" -ef "$EXPANSION_FACTOR" -e "$EPOCHS"

EXPANSION_FACTOR=4
uv run /BS/disentanglement/work/Disentanglement/MSAE/train.py -dt "$DT" -ds "$DS" -m "$M" -a "$ACTIVATION" -ef "$EXPANSION_FACTOR" -e "$EPOCHS" -bs $BATCH_SIZE --train_tc --lr "$LR"
#uv run /BS/disentanglement/work/Disentanglement/MSAE/train.py -dt "$DT" -ds "$DS" -m "$M" -a "$ACTIVATION" -ef "$EXPANSION_FACTOR" -e "$EPOCHS"

EXPANSION_FACTOR=8
uv run /BS/disentanglement/work/Disentanglement/MSAE/train.py -dt "$DT" -ds "$DS" -m "$M" -a "$ACTIVATION" -ef "$EXPANSION_FACTOR" -e "$EPOCHS" -bs $BATCH_SIZE --train_tc --lr "$LR"
# #uv run /BS/disentanglement/work/Disentanglement/MSAE/train.py -dt "$DT" -ds "$DS" -m "$M" -a "$ACTIVATION" -ef "$EXPANSION_FACTOR" -e "$EPOCHS"


M="ReLUTC" # "TopKSAE", "BatchTopKSAE"
ACTIVATION='ReLU_003' # 'TopKReLU_64'
LR=0.00005
EXPANSION_FACTOR=1
uv run /BS/disentanglement/work/Disentanglement/MSAE/train.py -dt "$DT" -ds "$DS" -m "$M" -a "$ACTIVATION" -ef "$EXPANSION_FACTOR" -e "$EPOCHS" -bs $BATCH_SIZE --train_tc --lr "$LR"
#uv run /BS/disentanglement/work/Disentanglement/MSAE/train.py -dt "$DT" -ds "$DS" -m "$M" -a "$ACTIVATION" -ef "$EXPANSION_FACTOR" -e "$EPOCHS"

EXPANSION_FACTOR=2
uv run /BS/disentanglement/work/Disentanglement/MSAE/train.py -dt "$DT" -ds "$DS" -m "$M" -a "$ACTIVATION" -ef "$EXPANSION_FACTOR" -e "$EPOCHS" -bs $BATCH_SIZE --train_tc --lr "$LR"
#uv run /BS/disentanglement/work/Disentanglement/MSAE/train.py -dt "$DT" -ds "$DS" -m "$M" -a "$ACTIVATION" -ef "$EXPANSION_FACTOR" -e "$EPOCHS"

EXPANSION_FACTOR=4
uv run /BS/disentanglement/work/Disentanglement/MSAE/train.py -dt "$DT" -ds "$DS" -m "$M" -a "$ACTIVATION" -ef "$EXPANSION_FACTOR" -e "$EPOCHS" -bs $BATCH_SIZE --train_tc --lr "$LR"
#uv run /BS/disentanglement/work/Disentanglement/MSAE/train.py -dt "$DT" -ds "$DS" -m "$M" -a "$ACTIVATION" -ef "$EXPANSION_FACTOR" -e "$EPOCHS"

EXPANSION_FACTOR=8
uv run /BS/disentanglement/work/Disentanglement/MSAE/train.py -dt "$DT" -ds "$DS" -m "$M" -a "$ACTIVATION" -ef "$EXPANSION_FACTOR" -e "$EPOCHS" -bs $BATCH_SIZE --train_tc --lr "$LR"



M="ReLUTC" # "TopKSAE", "BatchTopKSAE"
ACTIVATION='ReLU_001' # 'TopKReLU_64'
LR=0.0002
EXPANSION_FACTOR=1
uv run /BS/disentanglement/work/Disentanglement/MSAE/train.py -dt "$DT" -ds "$DS" -m "$M" -a "$ACTIVATION" -ef "$EXPANSION_FACTOR" -e "$EPOCHS" -bs $BATCH_SIZE --train_tc --lr "$LR"
#uv run /BS/disentanglement/work/Disentanglement/MSAE/train.py -dt "$DT" -ds "$DS" -m "$M" -a "$ACTIVATION" -ef "$EXPANSION_FACTOR" -e "$EPOCHS"

EXPANSION_FACTOR=2
uv run /BS/disentanglement/work/Disentanglement/MSAE/train.py -dt "$DT" -ds "$DS" -m "$M" -a "$ACTIVATION" -ef "$EXPANSION_FACTOR" -e "$EPOCHS" -bs $BATCH_SIZE --train_tc --lr "$LR"
#uv run /BS/disentanglement/work/Disentanglement/MSAE/train.py -dt "$DT" -ds "$DS" -m "$M" -a "$ACTIVATION" -ef "$EXPANSION_FACTOR" -e "$EPOCHS"

EXPANSION_FACTOR=4
uv run /BS/disentanglement/work/Disentanglement/MSAE/train.py -dt "$DT" -ds "$DS" -m "$M" -a "$ACTIVATION" -ef "$EXPANSION_FACTOR" -e "$EPOCHS" -bs $BATCH_SIZE --train_tc --lr "$LR"
#uv run /BS/disentanglement/work/Disentanglement/MSAE/train.py -dt "$DT" -ds "$DS" -m "$M" -a "$ACTIVATION" -ef "$EXPANSION_FACTOR" -e "$EPOCHS"

EXPANSION_FACTOR=8
uv run /BS/disentanglement/work/Disentanglement/MSAE/train.py -dt "$DT" -ds "$DS" -m "$M" -a "$ACTIVATION" -ef "$EXPANSION_FACTOR" -e "$EPOCHS" -bs $BATCH_SIZE --train_tc --lr "$LR"
# #uv run /BS/disentanglement/work/Disentanglement/MSAE/train.py -dt "$DT" -ds "$DS" -m "$M" -a "$ACTIVATION" -ef "$EXPANSION_FACTOR" -e "$EPOCHS"


M="ReLUTC" # "TopKSAE", "BatchTopKSAE"
ACTIVATION='ReLU_003' # 'TopKReLU_64'
LR=0.0002
EXPANSION_FACTOR=1
uv run /BS/disentanglement/work/Disentanglement/MSAE/train.py -dt "$DT" -ds "$DS" -m "$M" -a "$ACTIVATION" -ef "$EXPANSION_FACTOR" -e "$EPOCHS" -bs $BATCH_SIZE --train_tc --lr "$LR"
#uv run /BS/disentanglement/work/Disentanglement/MSAE/train.py -dt "$DT" -ds "$DS" -m "$M" -a "$ACTIVATION" -ef "$EXPANSION_FACTOR" -e "$EPOCHS"

EXPANSION_FACTOR=2
uv run /BS/disentanglement/work/Disentanglement/MSAE/train.py -dt "$DT" -ds "$DS" -m "$M" -a "$ACTIVATION" -ef "$EXPANSION_FACTOR" -e "$EPOCHS" -bs $BATCH_SIZE --train_tc --lr "$LR"
#uv run /BS/disentanglement/work/Disentanglement/MSAE/train.py -dt "$DT" -ds "$DS" -m "$M" -a "$ACTIVATION" -ef "$EXPANSION_FACTOR" -e "$EPOCHS"

EXPANSION_FACTOR=4
uv run /BS/disentanglement/work/Disentanglement/MSAE/train.py -dt "$DT" -ds "$DS" -m "$M" -a "$ACTIVATION" -ef "$EXPANSION_FACTOR" -e "$EPOCHS" -bs $BATCH_SIZE --train_tc --lr "$LR"
#uv run /BS/disentanglement/work/Disentanglement/MSAE/train.py -dt "$DT" -ds "$DS" -m "$M" -a "$ACTIVATION" -ef "$EXPANSION_FACTOR" -e "$EPOCHS"

EXPANSION_FACTOR=8
uv run /BS/disentanglement/work/Disentanglement/MSAE/train.py -dt "$DT" -ds "$DS" -m "$M" -a "$ACTIVATION" -ef "$EXPANSION_FACTOR" -e "$EPOCHS" -bs $BATCH_SIZE --train_tc --lr "$LR"

M="ReLUTC" # "TopKSAE", "BatchTopKSAE"
ACTIVATION='ReLU_001' # 'TopKReLU_64'
LR=0.02
EXPANSION_FACTOR=1
uv run /BS/disentanglement/work/Disentanglement/MSAE/train.py -dt "$DT" -ds "$DS" -m "$M" -a "$ACTIVATION" -ef "$EXPANSION_FACTOR" -e "$EPOCHS" -bs $BATCH_SIZE --train_tc --lr "$LR"
#uv run /BS/disentanglement/work/Disentanglement/MSAE/train.py -dt "$DT" -ds "$DS" -m "$M" -a "$ACTIVATION" -ef "$EXPANSION_FACTOR" -e "$EPOCHS"

EXPANSION_FACTOR=2
uv run /BS/disentanglement/work/Disentanglement/MSAE/train.py -dt "$DT" -ds "$DS" -m "$M" -a "$ACTIVATION" -ef "$EXPANSION_FACTOR" -e "$EPOCHS" -bs $BATCH_SIZE --train_tc --lr "$LR"
#uv run /BS/disentanglement/work/Disentanglement/MSAE/train.py -dt "$DT" -ds "$DS" -m "$M" -a "$ACTIVATION" -ef "$EXPANSION_FACTOR" -e "$EPOCHS"

EXPANSION_FACTOR=4
uv run /BS/disentanglement/work/Disentanglement/MSAE/train.py -dt "$DT" -ds "$DS" -m "$M" -a "$ACTIVATION" -ef "$EXPANSION_FACTOR" -e "$EPOCHS" -bs $BATCH_SIZE --train_tc --lr "$LR"
#uv run /BS/disentanglement/work/Disentanglement/MSAE/train.py -dt "$DT" -ds "$DS" -m "$M" -a "$ACTIVATION" -ef "$EXPANSION_FACTOR" -e "$EPOCHS"

EXPANSION_FACTOR=8
uv run /BS/disentanglement/work/Disentanglement/MSAE/train.py -dt "$DT" -ds "$DS" -m "$M" -a "$ACTIVATION" -ef "$EXPANSION_FACTOR" -e "$EPOCHS" -bs $BATCH_SIZE --train_tc --lr "$LR"

M="ReLUTC" # "TopKSAE", "BatchTopKSAE"
ACTIVATION='ReLU_003' # 'TopKReLU_64'
LR=0.02
EXPANSION_FACTOR=1
uv run /BS/disentanglement/work/Disentanglement/MSAE/train.py -dt "$DT" -ds "$DS" -m "$M" -a "$ACTIVATION" -ef "$EXPANSION_FACTOR" -e "$EPOCHS" -bs $BATCH_SIZE --train_tc --lr "$LR"
#uv run /BS/disentanglement/work/Disentanglement/MSAE/train.py -dt "$DT" -ds "$DS" -m "$M" -a "$ACTIVATION" -ef "$EXPANSION_FACTOR" -e "$EPOCHS"

EXPANSION_FACTOR=2
uv run /BS/disentanglement/work/Disentanglement/MSAE/train.py -dt "$DT" -ds "$DS" -m "$M" -a "$ACTIVATION" -ef "$EXPANSION_FACTOR" -e "$EPOCHS" -bs $BATCH_SIZE --train_tc --lr "$LR"
#uv run /BS/disentanglement/work/Disentanglement/MSAE/train.py -dt "$DT" -ds "$DS" -m "$M" -a "$ACTIVATION" -ef "$EXPANSION_FACTOR" -e "$EPOCHS"

EXPANSION_FACTOR=4
uv run /BS/disentanglement/work/Disentanglement/MSAE/train.py -dt "$DT" -ds "$DS" -m "$M" -a "$ACTIVATION" -ef "$EXPANSION_FACTOR" -e "$EPOCHS" -bs $BATCH_SIZE --train_tc --lr "$LR"
#uv run /BS/disentanglement/work/Disentanglement/MSAE/train.py -dt "$DT" -ds "$DS" -m "$M" -a "$ACTIVATION" -ef "$EXPANSION_FACTOR" -e "$EPOCHS"

EXPANSION_FACTOR=8
uv run /BS/disentanglement/work/Disentanglement/MSAE/train.py -dt "$DT" -ds "$DS" -m "$M" -a "$ACTIVATION" -ef "$EXPANSION_FACTOR" -e "$EPOCHS" -bs $BATCH_SIZE --train_tc --lr "$LR"

M="ReLUTC" # "TopKSAE", "BatchTopKSAE"
ACTIVATION='ReLU_001' # 'TopKReLU_64'
LR=0.005
EXPANSION_FACTOR=1
uv run /BS/disentanglement/work/Disentanglement/MSAE/train.py -dt "$DT" -ds "$DS" -m "$M" -a "$ACTIVATION" -ef "$EXPANSION_FACTOR" -e "$EPOCHS" -bs $BATCH_SIZE --train_tc --lr "$LR"
#uv run /BS/disentanglement/work/Disentanglement/MSAE/train.py -dt "$DT" -ds "$DS" -m "$M" -a "$ACTIVATION" -ef "$EXPANSION_FACTOR" -e "$EPOCHS"

EXPANSION_FACTOR=2
uv run /BS/disentanglement/work/Disentanglement/MSAE/train.py -dt "$DT" -ds "$DS" -m "$M" -a "$ACTIVATION" -ef "$EXPANSION_FACTOR" -e "$EPOCHS" -bs $BATCH_SIZE --train_tc --lr "$LR"
#uv run /BS/disentanglement/work/Disentanglement/MSAE/train.py -dt "$DT" -ds "$DS" -m "$M" -a "$ACTIVATION" -ef "$EXPANSION_FACTOR" -e "$EPOCHS"

EXPANSION_FACTOR=4
uv run /BS/disentanglement/work/Disentanglement/MSAE/train.py -dt "$DT" -ds "$DS" -m "$M" -a "$ACTIVATION" -ef "$EXPANSION_FACTOR" -e "$EPOCHS" -bs $BATCH_SIZE --train_tc --lr "$LR"
#uv run /BS/disentanglement/work/Disentanglement/MSAE/train.py -dt "$DT" -ds "$DS" -m "$M" -a "$ACTIVATION" -ef "$EXPANSION_FACTOR" -e "$EPOCHS"

EXPANSION_FACTOR=8
uv run /BS/disentanglement/work/Disentanglement/MSAE/train.py -dt "$DT" -ds "$DS" -m "$M" -a "$ACTIVATION" -ef "$EXPANSION_FACTOR" -e "$EPOCHS" -bs $BATCH_SIZE --train_tc --lr "$LR"

M="ReLUTC" # "TopKSAE", "BatchTopKSAE"
ACTIVATION='ReLU_003' # 'TopKReLU_64'
LR=0.005
EXPANSION_FACTOR=1
uv run /BS/disentanglement/work/Disentanglement/MSAE/train.py -dt "$DT" -ds "$DS" -m "$M" -a "$ACTIVATION" -ef "$EXPANSION_FACTOR" -e "$EPOCHS" -bs $BATCH_SIZE --train_tc --lr "$LR"
#uv run /BS/disentanglement/work/Disentanglement/MSAE/train.py -dt "$DT" -ds "$DS" -m "$M" -a "$ACTIVATION" -ef "$EXPANSION_FACTOR" -e "$EPOCHS"

EXPANSION_FACTOR=2
uv run /BS/disentanglement/work/Disentanglement/MSAE/train.py -dt "$DT" -ds "$DS" -m "$M" -a "$ACTIVATION" -ef "$EXPANSION_FACTOR" -e "$EPOCHS" -bs $BATCH_SIZE --train_tc --lr "$LR"
#uv run /BS/disentanglement/work/Disentanglement/MSAE/train.py -dt "$DT" -ds "$DS" -m "$M" -a "$ACTIVATION" -ef "$EXPANSION_FACTOR" -e "$EPOCHS"

EXPANSION_FACTOR=4
uv run /BS/disentanglement/work/Disentanglement/MSAE/train.py -dt "$DT" -ds "$DS" -m "$M" -a "$ACTIVATION" -ef "$EXPANSION_FACTOR" -e "$EPOCHS" -bs $BATCH_SIZE --train_tc --lr "$LR"
#uv run /BS/disentanglement/work/Disentanglement/MSAE/train.py -dt "$DT" -ds "$DS" -m "$M" -a "$ACTIVATION" -ef "$EXPANSION_FACTOR" -e "$EPOCHS"

EXPANSION_FACTOR=8
uv run /BS/disentanglement/work/Disentanglement/MSAE/train.py -dt "$DT" -ds "$DS" -m "$M" -a "$ACTIVATION" -ef "$EXPANSION_FACTOR" -e "$EPOCHS" -bs $BATCH_SIZE --train_tc --lr "$LR"

M="ReLUTC" # "TopKSAE", "BatchTopKSAE"
ACTIVATION='ReLU_001' # 'TopKReLU_64'
LR=0.001
EXPANSION_FACTOR=1
uv run /BS/disentanglement/work/Disentanglement/MSAE/train.py -dt "$DT" -ds "$DS" -m "$M" -a "$ACTIVATION" -ef "$EXPANSION_FACTOR" -e "$EPOCHS" -bs $BATCH_SIZE --train_tc --lr "$LR"
#uv run /BS/disentanglement/work/Disentanglement/MSAE/train.py -dt "$DT" -ds "$DS" -m "$M" -a "$ACTIVATION" -ef "$EXPANSION_FACTOR" -e "$EPOCHS"

EXPANSION_FACTOR=2
uv run /BS/disentanglement/work/Disentanglement/MSAE/train.py -dt "$DT" -ds "$DS" -m "$M" -a "$ACTIVATION" -ef "$EXPANSION_FACTOR" -e "$EPOCHS" -bs $BATCH_SIZE --train_tc --lr "$LR"
#uv run /BS/disentanglement/work/Disentanglement/MSAE/train.py -dt "$DT" -ds "$DS" -m "$M" -a "$ACTIVATION" -ef "$EXPANSION_FACTOR" -e "$EPOCHS"

EXPANSION_FACTOR=4
uv run /BS/disentanglement/work/Disentanglement/MSAE/train.py -dt "$DT" -ds "$DS" -m "$M" -a "$ACTIVATION" -ef "$EXPANSION_FACTOR" -e "$EPOCHS" -bs $BATCH_SIZE --train_tc --lr "$LR"
#uv run /BS/disentanglement/work/Disentanglement/MSAE/train.py -dt "$DT" -ds "$DS" -m "$M" -a "$ACTIVATION" -ef "$EXPANSION_FACTOR" -e "$EPOCHS"

EXPANSION_FACTOR=8
uv run /BS/disentanglement/work/Disentanglement/MSAE/train.py -dt "$DT" -ds "$DS" -m "$M" -a "$ACTIVATION" -ef "$EXPANSION_FACTOR" -e "$EPOCHS" -bs $BATCH_SIZE --train_tc --lr "$LR"

M="ReLUTC" # "TopKSAE", "BatchTopKSAE"
ACTIVATION='ReLU_003' # 'TopKReLU_64'
LR=0.001
EXPANSION_FACTOR=1
uv run /BS/disentanglement/work/Disentanglement/MSAE/train.py -dt "$DT" -ds "$DS" -m "$M" -a "$ACTIVATION" -ef "$EXPANSION_FACTOR" -e "$EPOCHS" -bs $BATCH_SIZE --train_tc --lr "$LR"
#uv run /BS/disentanglement/work/Disentanglement/MSAE/train.py -dt "$DT" -ds "$DS" -m "$M" -a "$ACTIVATION" -ef "$EXPANSION_FACTOR" -e "$EPOCHS"

EXPANSION_FACTOR=2
uv run /BS/disentanglement/work/Disentanglement/MSAE/train.py -dt "$DT" -ds "$DS" -m "$M" -a "$ACTIVATION" -ef "$EXPANSION_FACTOR" -e "$EPOCHS" -bs $BATCH_SIZE --train_tc --lr "$LR"
#uv run /BS/disentanglement/work/Disentanglement/MSAE/train.py -dt "$DT" -ds "$DS" -m "$M" -a "$ACTIVATION" -ef "$EXPANSION_FACTOR" -e "$EPOCHS"

EXPANSION_FACTOR=4
uv run /BS/disentanglement/work/Disentanglement/MSAE/train.py -dt "$DT" -ds "$DS" -m "$M" -a "$ACTIVATION" -ef "$EXPANSION_FACTOR" -e "$EPOCHS" -bs $BATCH_SIZE --train_tc --lr "$LR"
#uv run /BS/disentanglement/work/Disentanglement/MSAE/train.py -dt "$DT" -ds "$DS" -m "$M" -a "$ACTIVATION" -ef "$EXPANSION_FACTOR" -e "$EPOCHS"

EXPANSION_FACTOR=8
uv run /BS/disentanglement/work/Disentanglement/MSAE/train.py -dt "$DT" -ds "$DS" -m "$M" -a "$ACTIVATION" -ef "$EXPANSION_FACTOR" -e "$EPOCHS" -bs $BATCH_SIZE --train_tc --lr "$LR"


# EXPANSION_FACTOR=8
# uv run /BS/disentanglement/work/Disentanglement/MSAE/train.py -dt "$DT" -ds "$DS" -m "$M" -a "$ACTIVATION" -ef "$EXPANSION_FACTOR" -e "$EPOCHS" -bs $BATCH_SIZE --train_tc --lr $LR
# #uv run /BS/disentanglement/work/Disentanglement/MSAE/train.py -dt "$DT" -ds "$DS" -m "$M" -a "$ACTIVATION" -ef "$EXPANSION_FACTOR" -e "$EPOCHS"

# EXPANSION_FACTOR=16
# uv run /BS/disentanglement/work/Disentanglement/MSAE/train.py -dt "$DT" -ds "$DS" -m "$M" -a "$ACTIVATION" -ef "$EXPANSION_FACTOR" -e "$EPOCHS" -bs $BATCH_SIZE --train_tc --lr $LR
# #uv run /BS/disentanglement/work/Disentanglement/MSAE/train.py -dt "$DT" -ds "$DS" -m "$M" -a "$ACTIVATION" -ef "$EXPANSION_FACTOR" -e "$EPOCHS"

# EXPANSION_FACTOR=32
# uv run /BS/disentanglement/work/Disentanglement/MSAE/train.py -dt "$DT" -ds "$DS" -m "$M" -a "$ACTIVATION" -ef "$EXPANSION_FACTOR" -e "$EPOCHS" -bs $BATCH_SIZE --train_tc --lr $LR
# #uv run /BS/disentanglement/work/Disentanglement/MSAE/train.py -dt "$DT" -ds "$DS" -m "$M" -a "$ACTIVATION" -ef "$EXPANSION_FACTOR" -e "$EPOCHS"
