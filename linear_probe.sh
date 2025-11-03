#!/bin/bash
#SBATCH -p gpu16
#SBATCH --job-name=linear_probe
#SBATCH -o /BS/disentanglement/work/Disentanglement/jobs/sae_training/linear_eval-%j.out
#SBATCH --time=10:00:00
#SBATCH --gres gpu:1

STORE_DIR="/BS/disentanglement/work/sae/extract_embeddings"


## DINOv2
TRAIN_DATA="/BS/disentanglement/work/sae/linear_probe/imagenet_dinov2_vitl14_embeddings_train.npy"
DATA="/BS/disentanglement/work/sae/linear_probe/imagenet_dinov2_vitl14_embeddings_val.npy"
TRAIN_LABELS="/BS/disentanglement/work/sae/linear_probe/imagenet_dinov2_vitl14_labels_train.npy"
DATA_LABELS="/BS/disentanglement/work/sae/linear_probe/imagenet_dinov2_vitl14_labels_val.npy"

# #ReLU
# MODEL="/BS/disentanglement/work/msae/8192_1024_ReLU_003_False_False_0.0_imagenet_dinov2_vitl14_embeddings_train.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/linear_eval.py -m $MODEL -d $TRAIN_DATA -o $DATA_LABELS -t $TRAIN_LABELS -e $DATA 

# MODEL="/BS/disentanglement/work/msae/16384_1024_ReLU_003_False_False_0.0_imagenet_dinov2_vitl14_embeddings_train.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/linear_eval.py -m $MODEL -d $TRAIN_DATA -o $DATA_LABELS -t $TRAIN_LABELS -e $DATA 

# MODEL="/BS/disentanglement/work/msae/32768_1024_ReLU_003_False_False_0.0_imagenet_dinov2_vitl14_embeddings_train.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/linear_eval.py -m $MODEL -d $TRAIN_DATA -o $DATA_LABELS -t $TRAIN_LABELS -e $DATA 

# #Top256
# MODEL="/BS/disentanglement/work/msae/8192_1024_TopKReLU_256_False_False_0.0_imagenet_dinov2_vitl14_embeddings_train.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/linear_eval.py -m $MODEL -d $TRAIN_DATA -o $DATA_LABELS -t $TRAIN_LABELS -e $DATA 

# MODEL="/BS/disentanglement/work/msae/16384_1024_TopKReLU_256_False_False_0.0_imagenet_dinov2_vitl14_embeddings_train.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/linear_eval.py -m $MODEL -d $TRAIN_DATA -o $DATA_LABELS -t $TRAIN_LABELS -e $DATA 

# MODEL="/BS/disentanglement/work/msae/32768_1024_TopKReLU_256_False_False_0.0_imagenet_dinov2_vitl14_embeddings_train.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/linear_eval.py -m $MODEL -d $TRAIN_DATA -o $DATA_LABELS -t $TRAIN_LABELS -e $DATA 

# #Top64
# MODEL="/BS/disentanglement/work/msae/8192_1024_TopKReLU_64_False_False_0.0_imagenet_dinov2_vitl14_embeddings_train.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/linear_eval.py -m $MODEL -d $TRAIN_DATA -o $DATA_LABELS -t $TRAIN_LABELS -e $DATA 

# MODEL="/BS/disentanglement/work/msae/16384_1024_TopKReLU_64_False_False_0.0_imagenet_dinov2_vitl14_embeddings_train.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/linear_eval.py -m $MODEL -d $TRAIN_DATA -o $DATA_LABELS -t $TRAIN_LABELS -e $DATA 

# MODEL="/BS/disentanglement/work/msae/32768_1024_TopKReLU_64_False_False_0.0_imagenet_dinov2_vitl14_embeddings_train.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/linear_eval.py -m $MODEL -d $TRAIN_DATA -o $DATA_LABELS -t $TRAIN_LABELS -e $DATA 

# #BatchTop256
# MODEL="/BS/disentanglement/work/msae/batchtopk/8192_1024_TopKReLU_256_False_False_0.0_imagenet_dinov2_vitl14_embeddings_train.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/linear_eval.py -m $MODEL -d $TRAIN_DATA -o $DATA_LABELS -t $TRAIN_LABELS -e $DATA 

# MODEL="/BS/disentanglement/work/msae/batchtopk/16384_1024_TopKReLU_256_False_False_0.0_imagenet_dinov2_vitl14_embeddings_train.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/linear_eval.py -m $MODEL -d $TRAIN_DATA -o $DATA_LABELS -t $TRAIN_LABELS -e $DATA 

# MODEL="/BS/disentanglement/work/msae/batchtopk/32768_1024_TopKReLU_256_False_False_0.0_imagenet_dinov2_vitl14_embeddings_train.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/linear_eval.py -m $MODEL -d $TRAIN_DATA -o $DATA_LABELS -t $TRAIN_LABELS -e $DATA 

# #BatchTop64
# MODEL="/BS/disentanglement/work/msae/batchtopk/8192_1024_TopKReLU_64_False_False_0.0_imagenet_dinov2_vitl14_embeddings_train.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/linear_eval.py -m $MODEL -d $TRAIN_DATA -o $DATA_LABELS -t $TRAIN_LABELS -e $DATA 

# MODEL="/BS/disentanglement/work/msae/batchtopk/16384_1024_TopKReLU_64_False_False_0.0_imagenet_dinov2_vitl14_embeddings_train.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/linear_eval.py -m $MODEL -d $TRAIN_DATA -o $DATA_LABELS -t $TRAIN_LABELS -e $DATA 

# MODEL="/BS/disentanglement/work/msae/batchtopk/32768_1024_TopKReLU_64_False_False_0.0_imagenet_dinov2_vitl14_embeddings_train.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/linear_eval.py -m $MODEL -d $TRAIN_DATA -o $DATA_LABELS -t $TRAIN_LABELS -e $DATA 

# #MSAE RW
# MODEL="/BS/disentanglement/work/msae/8192_1024_TopKReLU_64_RW_False_False_0.0_imagenet_dinov2_vitl14_embeddings_train.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/linear_eval.py -m $MODEL -d $TRAIN_DATA -o $DATA_LABELS -t $TRAIN_LABELS -e $DATA 

# MODEL="/BS/disentanglement/work/msae/16384_1024_TopKReLU_64_RW_False_False_0.0_imagenet_dinov2_vitl14_embeddings_train.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/linear_eval.py -m $MODEL -d $TRAIN_DATA -o $DATA_LABELS -t $TRAIN_LABELS -e $DATA 

# MODEL="/BS/disentanglement/work/msae/32768_1024_TopKReLU_64_RW_False_False_0.0_imagenet_dinov2_vitl14_embeddings_train.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/linear_eval.py -m $MODEL -d $TRAIN_DATA -o $DATA_LABELS -t $TRAIN_LABELS -e $DATA 

# #MSAE UW
# MODEL="/BS/disentanglement/work/msae/8192_1024_TopKReLU_64_UW_False_False_0.0_imagenet_dinov2_vitl14_embeddings_train.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/linear_eval.py -m $MODEL -d $TRAIN_DATA -o $DATA_LABELS -t $TRAIN_LABELS -e $DATA 

# MODEL="/BS/disentanglement/work/msae/16384_1024_TopKReLU_64_UW_False_False_0.0_imagenet_dinov2_vitl14_embeddings_train.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/linear_eval.py -m $MODEL -d $TRAIN_DATA -o $DATA_LABELS -t $TRAIN_LABELS -e $DATA 

# MODEL="/BS/disentanglement/work/msae/32768_1024_TopKReLU_64_UW_False_False_0.0_imagenet_dinov2_vitl14_embeddings_train.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/linear_eval.py -m $MODEL -d $TRAIN_DATA -o $DATA_LABELS -t $TRAIN_LABELS -e $DATA 

# echo "DINOv2 evaluated"

# ##ViTB16
# TRAIN_DATA="/BS/disentanglement/work/sae/linear_probe/imagenet_vit_b_16_embeddings_train.npy"
# DATA="/BS/disentanglement/work/sae/linear_probe/imagenet_vit_b_16_embeddings_val.npy"
# TRAIN_LABELS="/BS/disentanglement/work/sae/linear_probe/imagenet_vit_b_16_labels_train.npy"
# DATA_LABELS="/BS/disentanglement/work/sae/linear_probe/imagenet_vit_b_16_labels_val.npy"

# #ReLU 
# MODEL="/BS/disentanglement/work/msae/6144_768_ReLU_003_False_False_0.0_imagenet_vit_b_16_embeddings_train.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/linear_eval.py -m $MODEL -d $TRAIN_DATA -o $DATA_LABELS -t $TRAIN_LABELS -e $DATA 

# MODEL="/BS/disentanglement/work/msae/12288_768_ReLU_003_False_False_0.0_imagenet_vit_b_16_embeddings_train.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/linear_eval.py -m $MODEL -d $TRAIN_DATA -o $DATA_LABELS -t $TRAIN_LABELS -e $DATA 

# MODEL="/BS/disentanglement/work/msae/24576_768_ReLU_003_False_False_0.0_imagenet_vit_b_16_embeddings_train.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/linear_eval.py -m $MODEL -d $TRAIN_DATA -o $DATA_LABELS -t $TRAIN_LABELS -e $DATA 

# #Top256
# MODEL="/BS/disentanglement/work/msae/6144_768_TopKReLU_256_False_False_0.0_imagenet_vit_b_16_embeddings_train.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/linear_eval.py -m $MODEL -d $TRAIN_DATA -o $DATA_LABELS -t $TRAIN_LABELS -e $DATA 

# MODEL="/BS/disentanglement/work/msae/12288_768_TopKReLU_256_False_False_0.0_imagenet_vit_b_16_embeddings_train.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/linear_eval.py -m $MODEL -d $TRAIN_DATA -o $DATA_LABELS -t $TRAIN_LABELS -e $DATA 

# MODEL="/BS/disentanglement/work/msae/24576_768_TopKReLU_256_False_False_0.0_imagenet_vit_b_16_embeddings_train.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/linear_eval.py -m $MODEL -d $TRAIN_DATA -o $DATA_LABELS -t $TRAIN_LABELS -e $DATA 

# #Top64
# MODEL="/BS/disentanglement/work/msae/6144_768_TopKReLU_64_False_False_0.0_imagenet_vit_b_16_embeddings_train.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/linear_eval.py -m $MODEL -d $TRAIN_DATA -o $DATA_LABELS -t $TRAIN_LABELS -e $DATA 

# MODEL="/BS/disentanglement/work/msae/12288_768_TopKReLU_64_False_False_0.0_imagenet_vit_b_16_embeddings_train.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/linear_eval.py -m $MODEL -d $TRAIN_DATA -o $DATA_LABELS -t $TRAIN_LABELS -e $DATA 

# MODEL="/BS/disentanglement/work/msae/24576_768_TopKReLU_64_False_False_0.0_imagenet_vit_b_16_embeddings_train.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/linear_eval.py -m $MODEL -d $TRAIN_DATA -o $DATA_LABELS -t $TRAIN_LABELS -e $DATA 

# #BatchTop256
# MODEL="/BS/disentanglement/work/msae/batchtopk/6144_768_TopKReLU_256_False_False_0.0_imagenet_vit_b_16_embeddings_train.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/linear_eval.py -m $MODEL -d $TRAIN_DATA -o $DATA_LABELS -t $TRAIN_LABELS -e $DATA 

# MODEL="/BS/disentanglement/work/msae/batchtopk/12288_768_TopKReLU_256_False_False_0.0_imagenet_vit_b_16_embeddings_train.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/linear_eval.py -m $MODEL -d $TRAIN_DATA -o $DATA_LABELS -t $TRAIN_LABELS -e $DATA 

# MODEL="/BS/disentanglement/work/msae/batchtopk/24576_768_TopKReLU_256_False_False_0.0_imagenet_vit_b_16_embeddings_train.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/linear_eval.py -m $MODEL -d $TRAIN_DATA -o $DATA_LABELS -t $TRAIN_LABELS -e $DATA 

# #BatchTop64
# MODEL="/BS/disentanglement/work/msae/batchtopk/6144_768_TopKReLU_64_False_False_0.0_imagenet_vit_b_16_embeddings_train.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/linear_eval.py -m $MODEL -d $TRAIN_DATA -o $DATA_LABELS -t $TRAIN_LABELS -e $DATA 

# MODEL="/BS/disentanglement/work/msae/batchtopk/12288_768_TopKReLU_64_False_False_0.0_imagenet_vit_b_16_embeddings_train.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/linear_eval.py -m $MODEL -d $TRAIN_DATA -o $DATA_LABELS -t $TRAIN_LABELS -e $DATA 

# MODEL="/BS/disentanglement/work/msae/batchtopk/24576_768_TopKReLU_64_False_False_0.0_imagenet_vit_b_16_embeddings_train.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/linear_eval.py -m $MODEL -d $TRAIN_DATA -o $DATA_LABELS -t $TRAIN_LABELS -e $DATA 

# #MSAE RW
# MODEL="/BS/disentanglement/work/msae/6144_768_TopKReLU_64_RW_False_False_0.0_imagenet_vit_b_16_embeddings_train.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/linear_eval.py -m $MODEL -d $TRAIN_DATA -o $DATA_LABELS -t $TRAIN_LABELS -e $DATA 

# MODEL="/BS/disentanglement/work/msae/12288_768_TopKReLU_64_RW_False_False_0.0_imagenet_vit_b_16_embeddings_train.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/linear_eval.py -m $MODEL -d $TRAIN_DATA -o $DATA_LABELS -t $TRAIN_LABELS -e $DATA 

# MODEL="/BS/disentanglement/work/msae/24576_768_TopKReLU_64_RW_False_False_0.0_imagenet_vit_b_16_embeddings_train.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/linear_eval.py -m $MODEL -d $TRAIN_DATA -o $DATA_LABELS -t $TRAIN_LABELS -e $DATA 

# #MSAE UW
# MODEL="/BS/disentanglement/work/msae/6144_768_TopKReLU_64_UW_False_False_0.0_imagenet_vit_b_16_embeddings_train.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/linear_eval.py -m $MODEL -d $TRAIN_DATA -o $DATA_LABELS -t $TRAIN_LABELS -e $DATA 

# MODEL="/BS/disentanglement/work/msae/12288_768_TopKReLU_64_UW_False_False_0.0_imagenet_vit_b_16_embeddings_train.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/linear_eval.py -m $MODEL -d $TRAIN_DATA -o $DATA_LABELS -t $TRAIN_LABELS -e $DATA 

# MODEL="/BS/disentanglement/work/msae/24576_768_TopKReLU_64_UW_False_False_0.0_imagenet_vit_b_16_embeddings_train.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/linear_eval.py -m $MODEL -d $TRAIN_DATA -o $DATA_LABELS -t $TRAIN_LABELS -e $DATA 


# echo "ViT-b/16 evaluated"

# ##ResNet50
TRAIN_DATA="/BS/disentanglement/work/sae/linear_probe/imagenet_resnet50_embeddings_train.npy"
DATA="/BS/disentanglement/work/sae/linear_probe/imagenet_resnet50_embeddings_val.npy"
TRAIN_LABELS="/BS/disentanglement/work/sae/linear_probe/imagenet_resnet50_labels_train.npy"
DATA_LABELS="/BS/disentanglement/work/sae/linear_probe/imagenet_resnet50_labels_val.npy"

# #ReLU
# MODEL="/BS/disentanglement/work/msae/16384_2048_ReLU_003_False_False_0.0_imagenet_resnet50_embeddings.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/linear_eval.py -m $MODEL -d $TRAIN_DATA -o $DATA_LABELS -t $TRAIN_LABELS -e $DATA 

# MODEL="/BS/disentanglement/work/msae/32768_2048_ReLU_003_False_False_0.0_imagenet_resnet50_embeddings.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/linear_eval.py -m $MODEL -d $TRAIN_DATA -o $DATA_LABELS -t $TRAIN_LABELS -e $DATA 

# MODEL="/BS/disentanglement/work/msae/65536_2048_ReLU_003_False_False_0.0_imagenet_resnet50_embeddings.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/linear_eval.py -m $MODEL -d $TRAIN_DATA -o $DATA_LABELS -t $TRAIN_LABELS -e $DATA 

# #Top256
# MODEL="/BS/disentanglement/work/msae/16384_2048_TopKReLU_256_False_False_0.0_imagenet_resnet50_embeddings.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/linear_eval.py -m $MODEL -d $TRAIN_DATA -o $DATA_LABELS -t $TRAIN_LABELS -e $DATA 

# MODEL="/BS/disentanglement/work/msae/32768_2048_TopKReLU_256_False_False_0.0_imagenet_resnet50_embeddings.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/linear_eval.py -m $MODEL -d $TRAIN_DATA -o $DATA_LABELS -t $TRAIN_LABELS -e $DATA 

# MODEL="/BS/disentanglement/work/msae/65536_2048_TopKReLU_256_False_False_0.0_imagenet_resnet50_embeddings.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/linear_eval.py -m $MODEL -d $TRAIN_DATA -o $DATA_LABELS -t $TRAIN_LABELS -e $DATA 

# #Top64
# MODEL="/BS/disentanglement/work/msae/16384_2048_TopKReLU_64_False_False_0.0_imagenet_resnet50_embeddings.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/linear_eval.py -m $MODEL -d $TRAIN_DATA -o $DATA_LABELS -t $TRAIN_LABELS -e $DATA 

# MODEL="/BS/disentanglement/work/msae/32768_2048_TopKReLU_64_False_False_0.0_imagenet_resnet50_embeddings.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/linear_eval.py -m $MODEL -d $TRAIN_DATA -o $DATA_LABELS -t $TRAIN_LABELS -e $DATA 

# MODEL="/BS/disentanglement/work/msae/65536_2048_TopKReLU_64_False_False_0.0_imagenet_resnet50_embeddings.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/linear_eval.py -m $MODEL -d $TRAIN_DATA -o $DATA_LABELS -t $TRAIN_LABELS -e $DATA 

# #BatchTop256
# MODEL="/BS/disentanglement/work/msae/batchtopk/16384_2048_TopKReLU_256_False_False_0.0_imagenet_resnet50_embeddings.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/linear_eval.py -m $MODEL -d $TRAIN_DATA -o $DATA_LABELS -t $TRAIN_LABELS -e $DATA 

# MODEL="/BS/disentanglement/work/msae/batchtopk/32768_2048_TopKReLU_256_False_False_0.0_imagenet_resnet50_embeddings.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/linear_eval.py -m $MODEL -d $TRAIN_DATA -o $DATA_LABELS -t $TRAIN_LABELS -e $DATA 

# MODEL="/BS/disentanglement/work/msae/batchtopk/65536_2048_TopKReLU_256_False_False_0.0_imagenet_resnet50_embeddings.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/linear_eval.py -m $MODEL -d $TRAIN_DATA -o $DATA_LABELS -t $TRAIN_LABELS -e $DATA 

# #BatchTop64
# MODEL="/BS/disentanglement/work/msae/batchtopk/16384_2048_TopKReLU_64_False_False_0.0_imagenet_resnet50_embeddings.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/linear_eval.py -m $MODEL -d $TRAIN_DATA -o $DATA_LABELS -t $TRAIN_LABELS -e $DATA 

# MODEL="/BS/disentanglement/work/msae/batchtopk/32768_2048_TopKReLU_64_False_False_0.0_imagenet_resnet50_embeddings.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/linear_eval.py -m $MODEL -d $TRAIN_DATA -o $DATA_LABELS -t $TRAIN_LABELS -e $DATA 

# MODEL="/BS/disentanglement/work/msae/batchtopk/65536_2048_TopKReLU_64_False_False_0.0_imagenet_resnet50_embeddings.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/linear_eval.py -m $MODEL -d $TRAIN_DATA -o $DATA_LABELS -t $TRAIN_LABELS -e $DATA 

# #MSAE RW
# MODEL="/BS/disentanglement/work/msae/16384_2048_TopKReLU_64_RW_False_False_0.0_imagenet_resnet50_embeddings.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/linear_eval.py -m $MODEL -d $TRAIN_DATA -o $DATA_LABELS -t $TRAIN_LABELS -e $DATA 

# MODEL="/BS/disentanglement/work/msae/32768_2048_TopKReLU_64_RW_False_False_0.0_imagenet_resnet50_embeddings.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/linear_eval.py -m $MODEL -d $TRAIN_DATA -o $DATA_LABELS -t $TRAIN_LABELS -e $DATA 

# MODEL="/BS/disentanglement/work/msae/65536_2048_TopKReLU_64_RW_False_False_0.0_imagenet_resnet50_embeddings.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/linear_eval.py -m $MODEL -d $TRAIN_DATA -o $DATA_LABELS -t $TRAIN_LABELS -e $DATA 

# #MSAE UW
# MODEL="/BS/disentanglement/work/msae/16384_2048_TopKReLU_64_UW_False_False_0.0_imagenet_resnet50_embeddings.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/linear_eval.py -m $MODEL -d $TRAIN_DATA -o $DATA_LABELS -t $TRAIN_LABELS -e $DATA 

# MODEL="/BS/disentanglement/work/msae/32768_2048_TopKReLU_64_UW_False_False_0.0_imagenet_resnet50_embeddings.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/linear_eval.py -m $MODEL -d $TRAIN_DATA -o $DATA_LABELS -t $TRAIN_LABELS -e $DATA 

MODEL="/BS/disentanglement/work/msae/65536_2048_TopKReLU_64_UW_False_False_0.0_imagenet_resnet50_embeddings.pth"
uv run /BS/disentanglement/work/Disentanglement/MSAE/linear_eval.py -m $MODEL -d $TRAIN_DATA -o $DATA_LABELS -t $TRAIN_LABELS -e $DATA 


echo "ResNet50 evaluated"


echo "STARTING HC EVALUATION"

## DINOv2
TRAIN_DATA="/BS/disentanglement/work/sae/linear_probe/imagenet_dinov2_vitl14_embeddings_train.npy"
DATA="/BS/disentanglement/work/sae/linear_probe/imagenet_dinov2_vitl14_embeddings_val.npy"
TRAIN_LABELS="/BS/disentanglement/work/sae/linear_probe/imagenet_dinov2_vitl14_labels_train.npy"
DATA_LABELS="/BS/disentanglement/work/sae/linear_probe/imagenet_dinov2_vitl14_labels_val.npy"

TAU=0.03
MODEL="/BS/disentanglement/work/sae/8192_1024_ReLU_003_False_False_0.0_imagenet_dinov2_vitl14_embeddings_train.pth"
#uv run /BS/disentanglement/work/Disentanglement/MSAE/linear_eval.py -m $MODEL -d $TRAIN_DATA -o $DATA_LABELS -t $TRAIN_LABELS -e $DATA --eval_hc True --tau $TAU

TAU=0.02
MODEL="/BS/disentanglement/work/sae/16384_1024_ReLU_003_False_False_0.0_imagenet_dinov2_vitl14_embeddings_train.pth"
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
#uv run /BS/disentanglement/work/Disentanglement/MSAE/linear_eval.py -m $MODEL -d $TRAIN_DATA -o $DATA_LABELS -t $TRAIN_LABELS -e $DATA --eval-hc True --tau $TAU

TAU=0.01
MODEL="/BS/disentanglement/work/sae/12288_768_ReLU_003_False_False_0.0_imagenet_vit_b_16_embeddings_train.pth"
#uv run /BS/disentanglement/work/Disentanglement/MSAE/linear_eval.py -m $MODEL -d $TRAIN_DATA -o $DATA_LABELS -t $TRAIN_LABELS -e $DATA --eval-hc True --tau $TAU


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
#uv run /BS/disentanglement/work/Disentanglement/MSAE/linear_eval.py -m $MODEL -d $TRAIN_DATA -o $DATA_LABELS -t $TRAIN_LABELS -e $DATA --eval-hc True --tau $TAU

echo "ResNet50 evaluated"