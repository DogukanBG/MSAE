#!/bin/sh
#SBATCH -p gpu16
#SBATCH -t 12:00:00
#SBATCH -o /BS/disentanglement/work/Disentanglement/jobs/sae_training/sae_eval-%j.out
#SBATCH --gres gpu:a40:1

STORE_DIR="/BS/disentanglement/work/sae/extract_embeddings"


## DINOv2
DATASET="/BS/disentanglement/work/sae/imagenet_dinov2_vitl14_embeddings_val.npy"

# #ReLU
# MODEL="/BS/disentanglement/work/msae/8192_1024_ReLU_003_False_False_0.0_imagenet_dinov2_vitl14_embeddings_train.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/extract_sae_embeddings.py -m $MODEL -d $DATASET -o $STORE_DIR

# MODEL="/BS/disentanglement/work/msae/16384_1024_ReLU_003_False_False_0.0_imagenet_dinov2_vitl14_embeddings_train.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/extract_sae_embeddings.py -m $MODEL -d $DATASET -o $STORE_DIR

# MODEL="/BS/disentanglement/work/msae/32768_1024_ReLU_003_False_False_0.0_imagenet_dinov2_vitl14_embeddings_train.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/extract_sae_embeddings.py -m $MODEL -d $DATASET -o $STORE_DIR

# #Top64
# MODEL="/BS/disentanglement/work/msae/8192_1024_TopKReLU_64_False_False_0.0_imagenet_dinov2_vitl14_embeddings_train.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/extract_sae_embeddings.py -m $MODEL -d $DATASET -o $STORE_DIR

# MODEL="/BS/disentanglement/work/msae/16384_1024_TopKReLU_64_False_False_0.0_imagenet_dinov2_vitl14_embeddings_train.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/extract_sae_embeddings.py -m $MODEL -d $DATASET -o $STORE_DIR

# MODEL="/BS/disentanglement/work/msae/32768_1024_TopKReLU_64_False_False_0.0_imagenet_dinov2_vitl14_embeddings_train.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/extract_sae_embeddings.py -m $MODEL -d $DATASET -o $STORE_DIR

# #Top256
# MODEL="/BS/disentanglement/work/msae/8192_1024_TopKReLU_256_False_False_0.0_imagenet_dinov2_vitl14_embeddings_train.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/extract_sae_embeddings.py -m $MODEL -d $DATASET -o $STORE_DIR

# MODEL="/BS/disentanglement/work/msae/16384_1024_TopKReLU_256_False_False_0.0_imagenet_dinov2_vitl14_embeddings_train.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/extract_sae_embeddings.py -m $MODEL -d $DATASET -o $STORE_DIR

# MODEL="/BS/disentanglement/work/msae/32768_1024_TopKReLU_256_False_False_0.0_imagenet_dinov2_vitl14_embeddings_train.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/extract_sae_embeddings.py -m $MODEL -d $DATASET -o $STORE_DIR


# #BatchTop64
# MODEL="/BS/disentanglement/work/msae/batchtopk/8192_1024_TopKReLU_64_False_False_0.0_imagenet_dinov2_vitl14_embeddings_train.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/extract_sae_embeddings.py -m $MODEL -d $DATASET -o $STORE_DIR

# MODEL="/BS/disentanglement/work/msae/batchtopk/16384_1024_TopKReLU_64_False_False_0.0_imagenet_dinov2_vitl14_embeddings_train.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/extract_sae_embeddings.py -m $MODEL -d $DATASET -o $STORE_DIR

# MODEL="/BS/disentanglement/work/msae/batchtopk/32768_1024_TopKReLU_64_False_False_0.0_imagenet_dinov2_vitl14_embeddings_train.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/extract_sae_embeddings.py -m $MODEL -d $DATASET -o $STORE_DIR

# #BatchTop256
# MODEL="/BS/disentanglement/work/msae/batchtopk/8192_1024_TopKReLU_256_False_False_0.0_imagenet_dinov2_vitl14_embeddings_train.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/extract_sae_embeddings.py -m $MODEL -d $DATASET -o $STORE_DIR

# MODEL="/BS/disentanglement/work/msae/batchtopk/16384_1024_TopKReLU_256_False_False_0.0_imagenet_dinov2_vitl14_embeddings_train.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/extract_sae_embeddings.py -m $MODEL -d $DATASET -o $STORE_DIR

# MODEL="/BS/disentanglement/work/msae/batchtopk/32768_1024_TopKReLU_256_False_False_0.0_imagenet_dinov2_vitl14_embeddings_train.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/extract_sae_embeddings.py -m $MODEL -d $DATASET -o $STORE_DIR

#MSAE RW
MODEL="/BS/disentanglement/work/msae/8192_1024_TopKReLU_64_RW_False_False_0.0_imagenet_dinov2_vitl14_embeddings_train.pth"
uv run /BS/disentanglement/work/Disentanglement/MSAE/extract_sae_embeddings.py -m $MODEL -d $DATASET -o $STORE_DIR

MODEL="/BS/disentanglement/work/msae/16384_1024_TopKReLU_64_RW_False_False_0.0_imagenet_dinov2_vitl14_embeddings_train.pth"
uv run /BS/disentanglement/work/Disentanglement/MSAE/extract_sae_embeddings.py -m $MODEL -d $DATASET -o $STORE_DIR

MODEL="/BS/disentanglement/work/msae/32768_1024_TopKReLU_64_RW_False_False_0.0_imagenet_dinov2_vitl14_embeddings_train.pth"
uv run /BS/disentanglement/work/Disentanglement/MSAE/extract_sae_embeddings.py -m $MODEL -d $DATASET -o $STORE_DIR

#MSAE UW
MODEL="/BS/disentanglement/work/msae/8192_1024_TopKReLU_64_UW_False_False_0.0_imagenet_dinov2_vitl14_embeddings_train.pth"
uv run /BS/disentanglement/work/Disentanglement/MSAE/extract_sae_embeddings.py -m $MODEL -d $DATASET -o $STORE_DIR

MODEL="/BS/disentanglement/work/msae/16384_1024_TopKReLU_64_UW_False_False_0.0_imagenet_dinov2_vitl14_embeddings_train.pth"
uv run /BS/disentanglement/work/Disentanglement/MSAE/extract_sae_embeddings.py -m $MODEL -d $DATASET -o $STORE_DIR

MODEL="/BS/disentanglement/work/msae/32768_1024_TopKReLU_64_UW_False_False_0.0_imagenet_dinov2_vitl14_embeddings_train.pth"
uv run /BS/disentanglement/work/Disentanglement/MSAE/extract_sae_embeddings.py -m $MODEL -d $DATASET -o $STORE_DIR

echo "DINOv2 evaluated"

# ##ViTB16
# DATASET="/BS/disentanglement/work/sae/imagenet_vit_b_16_embeddings_val.npy"
# #ReLU
# MODEL="/BS/disentanglement/work/msae/6144_768_ReLU_003_False_False_0.0_imagenet_vit_b_16_embeddings_train.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/extract_sae_embeddings.py -m $MODEL -d $DATASET -o $STORE_DIR

# MODEL="/BS/disentanglement/work/msae/12288_768_ReLU_003_False_False_0.0_imagenet_vit_b_16_embeddings_train.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/extract_sae_embeddings.py -m $MODEL -d $DATASET -o $STORE_DIR

# MODEL="/BS/disentanglement/work/msae/24576_768_ReLU_003_False_False_0.0_imagenet_vit_b_16_embeddings_train.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/extract_sae_embeddings.py -m $MODEL -d $DATASET -o $STORE_DIR

# #Top64
# MODEL="/BS/disentanglement/work/msae/6144_768_TopKReLU_64_False_False_0.0_imagenet_vit_b_16_embeddings_train.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/extract_sae_embeddings.py -m $MODEL -d $DATASET -o $STORE_DIR

# MODEL="/BS/disentanglement/work/msae/12288_768_TopKReLU_64_False_False_0.0_imagenet_vit_b_16_embeddings_train.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/extract_sae_embeddings.py -m $MODEL -d $DATASET -o $STORE_DIR

# MODEL="/BS/disentanglement/work/msae/24576_768_TopKReLU_64_False_False_0.0_imagenet_vit_b_16_embeddings_train.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/extract_sae_embeddings.py -m $MODEL -d $DATASET -o $STORE_DIR

# # #Top256
# MODEL="/BS/disentanglement/work/msae/6144_768_TopKReLU_256_False_False_0.0_imagenet_vit_b_16_embeddings_train.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/extract_sae_embeddings.py -m $MODEL -d $DATASET -o $STORE_DIR

# MODEL="/BS/disentanglement/work/msae/12288_768_TopKReLU_256_False_False_0.0_imagenet_vit_b_16_embeddings_train.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/extract_sae_embeddings.py -m $MODEL -d $DATASET -o $STORE_DIR

# MODEL="/BS/disentanglement/work/msae/24576_768_TopKReLU_256_False_False_0.0_imagenet_vit_b_16_embeddings_train.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/extract_sae_embeddings.py -m $MODEL -d $DATASET -o $STORE_DIR

# # #BatchTop64
# MODEL="/BS/disentanglement/work/msae/batchtopk/6144_768_TopKReLU_64_False_False_0.0_imagenet_vit_b_16_embeddings_train.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/extract_sae_embeddings.py -m $MODEL -d $DATASET -o $STORE_DIR

# MODEL="/BS/disentanglement/work/msae/batchtopk/12288_768_TopKReLU_64_False_False_0.0_imagenet_vit_b_16_embeddings_train.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/extract_sae_embeddings.py -m $MODEL -d $DATASET -o $STORE_DIR

# MODEL="/BS/disentanglement/work/msae/batchtopk/24576_768_TopKReLU_64_False_False_0.0_imagenet_vit_b_16_embeddings_train.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/extract_sae_embeddings.py -m $MODEL -d $DATASET -o $STORE_DIR

# # #BatchTop256
# MODEL="/BS/disentanglement/work/msae/batchtopk/6144_768_TopKReLU_256_False_False_0.0_imagenet_vit_b_16_embeddings_train.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/extract_sae_embeddings.py -m $MODEL -d $DATASET -o $STORE_DIR

# MODEL="/BS/disentanglement/work/msae/batchtopk/12288_768_TopKReLU_256_False_False_0.0_imagenet_vit_b_16_embeddings_train.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/extract_sae_embeddings.py -m $MODEL -d $DATASET -o $STORE_DIR

# MODEL="/BS/disentanglement/work/msae/batchtopk/24576_768_TopKReLU_256_False_False_0.0_imagenet_vit_b_16_embeddings_train.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/extract_sae_embeddings.py -m $MODEL -d $DATASET -o $STORE_DIR

# #MSAE RW
# MODEL="/BS/disentanglement/work/msae/6144_768_TopKReLU_64_RW_False_False_0.0_imagenet_vit_b_16_embeddings_train.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/extract_sae_embeddings.py -m $MODEL -d $DATASET -o $STORE_DIR

# MODEL="/BS/disentanglement/work/msae/12288_768_TopKReLU_64_RW_False_False_0.0_imagenet_vit_b_16_embeddings_train.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/extract_sae_embeddings.py -m $MODEL -d $DATASET -o $STORE_DIR

# MODEL="/BS/disentanglement/work/msae/24576_768_TopKReLU_64_RW_False_False_0.0_imagenet_vit_b_16_embeddings_train.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/extract_sae_embeddings.py -m $MODEL -d $DATASET -o $STORE_DIR

# # #MSAE UW
# MODEL="/BS/disentanglement/work/msae/6144_768_TopKReLU_64_UW_False_False_0.0_imagenet_vit_b_16_embeddings_train.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/extract_sae_embeddings.py -m $MODEL -d $DATASET -o $STORE_DIR

# MODEL="/BS/disentanglement/work/msae/12288_768_TopKReLU_64_UW_False_False_0.0_imagenet_vit_b_16_embeddings_train.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/extract_sae_embeddings.py -m $MODEL -d $DATASET -o $STORE_DIR

# MODEL="/BS/disentanglement/work/msae/24576_768_TopKReLU_64_UW_False_False_0.0_imagenet_vit_b_16_embeddings_train.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/extract_sae_embeddings.py -m $MODEL -d $DATASET -o $STORE_DIR

# echo "ViT-b/16 evaluated"

# ##ResNet50
# DATASET="/BS/disentanglement/work/sae/imagenet_resnet50_embeddings_val.npy"
# #ReLU
# MODEL="/BS/disentanglement/work/msae/16384_2048_ReLU_003_False_False_0.0_imagenet_resnet50_embeddings.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/extract_sae_embeddings.py -m $MODEL -d $DATASET -o $STORE_DIR

# MODEL="/BS/disentanglement/work/msae/32768_2048_ReLU_003_False_False_0.0_imagenet_resnet50_embeddings.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/extract_sae_embeddings.py -m $MODEL -d $DATASET -o $STORE_DIR

# MODEL="/BS/disentanglement/work/msae/65536_2048_ReLU_003_False_False_0.0_imagenet_resnet50_embeddings.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/extract_sae_embeddings.py -m $MODEL -d $DATASET -o $STORE_DIR

# # #Top64
# MODEL="/BS/disentanglement/work/msae/16384_2048_TopKReLU_64_False_False_0.0_imagenet_resnet50_embeddings.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/extract_sae_embeddings.py -m $MODEL -d $DATASET -o $STORE_DIR

# MODEL="/BS/disentanglement/work/msae/32768_2048_TopKReLU_64_False_False_0.0_imagenet_resnet50_embeddings.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/extract_sae_embeddings.py -m $MODEL -d $DATASET -o $STORE_DIR

# MODEL="/BS/disentanglement/work/msae/65536_2048_TopKReLU_64_False_False_0.0_imagenet_resnet50_embeddings.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/extract_sae_embeddings.py -m $MODEL -d $DATASET -o $STORE_DIR

# #Top256
# MODEL="/BS/disentanglement/work/msae/16384_2048_TopKReLU_256_False_False_0.0_imagenet_resnet50_embeddings.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/extract_sae_embeddings.py -m $MODEL -d $DATASET -o $STORE_DIR

# MODEL="/BS/disentanglement/work/msae/32768_2048_TopKReLU_256_False_False_0.0_imagenet_resnet50_embeddings.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/extract_sae_embeddings.py -m $MODEL -d $DATASET -o $STORE_DIR

# MODEL="/BS/disentanglement/work/msae/65536_2048_TopKReLU_256_False_False_0.0_imagenet_resnet50_embeddings.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/extract_sae_embeddings.py -m $MODEL -d $DATASET -o $STORE_DIR

# #BatchTop64
# MODEL="/BS/disentanglement/work/msae/batchtopk/16384_2048_TopKReLU_64_False_False_0.0_imagenet_resnet50_embeddings.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/extract_sae_embeddings.py -m $MODEL -d $DATASET -o $STORE_DIR

# MODEL="/BS/disentanglement/work/msae/batchtopk/32768_2048_TopKReLU_64_False_False_0.0_imagenet_resnet50_embeddings.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/extract_sae_embeddings.py -m $MODEL -d $DATASET -o $STORE_DIR

# MODEL="/BS/disentanglement/work/msae/batchtopk/65536_2048_TopKReLU_64_False_False_0.0_imagenet_resnet50_embeddings.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/extract_sae_embeddings.py -m $MODEL -d $DATASET -o $STORE_DIR

# #BatchTop256
# MODEL="/BS/disentanglement/work/msae/batchtopk/16384_2048_TopKReLU_256_False_False_0.0_imagenet_resnet50_embeddings.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/extract_sae_embeddings.py -m $MODEL -d $DATASET -o $STORE_DIR

# MODEL="/BS/disentanglement/work/msae/batchtopk/32768_2048_TopKReLU_256_False_False_0.0_imagenet_resnet50_embeddings.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/extract_sae_embeddings.py -m $MODEL -d $DATASET -o $STORE_DIR

# MODEL="/BS/disentanglement/work/msae/batchtopk/65536_2048_TopKReLU_256_False_False_0.0_imagenet_resnet50_embeddings.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/extract_sae_embeddings.py -m $MODEL -d $DATASET -o $STORE_DIR

#MSAE RW
# MODEL="/BS/disentanglement/work/msae/16384_2048_TopKReLU_64_RW_False_False_0.0_imagenet_resnet50_embeddings.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/extract_sae_embeddings.py -m $MODEL -d $DATASET -o $STORE_DIR

# MODEL="/BS/disentanglement/work/msae/32768_2048_TopKReLU_64_RW_False_False_0.0_imagenet_resnet50_embeddings.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/extract_sae_embeddings.py -m $MODEL -d $DATASET -o $STORE_DIR

# MODEL="/BS/disentanglement/work/msae/65536_2048_TopKReLU_64_RW_False_False_0.0_imagenet_resnet50_embeddings.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/extract_sae_embeddings.py -m $MODEL -d $DATASET -o $STORE_DIR

# #MSAE UW
# MODEL="/BS/disentanglement/work/msae/16384_2048_TopKReLU_64_UW_False_False_0.0_imagenet_resnet50_embeddings.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/extract_sae_embeddings.py -m $MODEL -d $DATASET -o $STORE_DIR

# MODEL="/BS/disentanglement/work/msae/32768_2048_TopKReLU_64_UW_False_False_0.0_imagenet_resnet50_embeddings.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/extract_sae_embeddings.py -m $MODEL -d $DATASET -o $STORE_DIR

# MODEL="/BS/disentanglement/work/msae/65536_2048_TopKReLU_64_UW_False_False_0.0_imagenet_resnet50_embeddings.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/extract_sae_embeddings.py -m $MODEL -d $DATASET -o $STORE_DIR


echo "ResNet50 evaluated"