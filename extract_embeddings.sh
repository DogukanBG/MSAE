#!/bin/sh
#SBATCH -p gpu20
#SBATCH -t 2:00:00
#SBATCH -o /BS/disentanglement/work/Disentanglement/jobs/sae_jobs/MSAE-eval_sae-%j.out
#SBATCH --gres gpu:1

DATA_DIR="/scratch/inf0/user/mparcham/ILSVRC2012/val_categorized"
DATASET="imagenet"
MODEL="vit_b_16" #"dinov2_vitl14"
SAVE_DIR="/BS/disentanglement/work"
BATCH_SIZE=4096

DT="/BS/disentanglement/work/sae/imagenet_vit_b_16_embeddings_train.npy"
DS="/BS/disentanglement/work/sae/imagenet_vit_b_16_embeddings_val.npy"
EPOCHS=10

#uv run /BS/disentanglement/work/Disentanglement/MSAE/precompute_activations.py --model "$MODEL" --dataset "$DATASET" --data-path "$DATA_DIR" --save-dir "$SAVE_DIR" --batch-size $BATCH_SIZE --split train

MODEL="dinov2_vitl14"
uv run /BS/disentanglement/work/Disentanglement/MSAE/precompute_activations.py --model "$MODEL" --dataset "$DATASET" --data-path "$DATA_DIR" --save-dir "$SAVE_DIR" --batch-size $BATCH_SIZE --split val

MODEL="resnet50"
#uv run /BS/disentanglement/work/Disentanglement/MSAE/precompute_activations.py --model "$MODEL" --dataset "$DATASET" --data-path "$DATA_DIR" --save-dir "$SAVE_DIR" --batch-size $BATCH_SIZE --split val

DATA_DIR="/scratch/inf0/user/mparcham/ILSVRC2012/train"
#uv run /BS/disentanglement/work/Disentanglement/MSAE/precompute_activations.py --model "$MODEL" --dataset "$DATASET" --data-path "$DATA_DIR" --save-dir "$SAVE_DIR" --batch-size $BATCH_SIZE --split train

MODEL="dinov2_vitl14"
uv run /BS/disentanglement/work/Disentanglement/MSAE/precompute_activations.py --model "$MODEL" --dataset "$DATASET" --data-path "$DATA_DIR" --save-dir "$SAVE_DIR" --batch-size $BATCH_SIZE --split train

MODEL="resnet50"
#uv run /BS/disentanglement/work/Disentanglement/MSAE/precompute_activations.py --model "$MODEL" --dataset "$DATASET" --data-path "$DATA_DIR" --save-dir "$SAVE_DIR" --batch-size $BATCH_SIZE --split train