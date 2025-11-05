#!/bin/bash
#SBATCH -p gpu24
#SBATCH -t 12:00:00
#SBATCH -o /BS/disentanglement/work/Disentanglement/jobs/ms_scores/vit_n%a.out
#SBATCH --gres gpu:1
#SBATCH --array=26-28%10


# Define all model-dataset combinations
declare -a CONFIGS=(
    # DINOv2 - ReLU
    "/BS/disentanglement/work/sae/imagenet_dinov2_vitl14_embeddings_val.npy:/BS/disentanglement/work/msae/8192_1024_ReLU_003_False_False_0.0_imagenet_dinov2_vitl14_embeddings_train.pth"
    "/BS/disentanglement/work/sae/imagenet_dinov2_vitl14_embeddings_val.npy:/BS/disentanglement/work/msae/16384_1024_ReLU_003_False_False_0.0_imagenet_dinov2_vitl14_embeddings_train.pth"
    "/BS/disentanglement/work/sae/imagenet_dinov2_vitl14_embeddings_val.npy:/BS/disentanglement/work/msae/32768_1024_ReLU_003_False_False_0.0_imagenet_dinov2_vitl14_embeddings_train.pth"
    # DINOv2 - Top64
    "/BS/disentanglement/work/sae/imagenet_dinov2_vitl14_embeddings_val.npy:/BS/disentanglement/work/msae/8192_1024_TopKReLU_64_False_False_0.0_imagenet_dinov2_vitl14_embeddings_train.pth"
    "/BS/disentanglement/work/sae/imagenet_dinov2_vitl14_embeddings_val.npy:/BS/disentanglement/work/msae/16384_1024_TopKReLU_64_False_False_0.0_imagenet_dinov2_vitl14_embeddings_train.pth"
    "/BS/disentanglement/work/sae/imagenet_dinov2_vitl14_embeddings_val.npy:/BS/disentanglement/work/msae/32768_1024_TopKReLU_64_False_False_0.0_imagenet_dinov2_vitl14_embeddings_train.pth"
    # DINOv2 - Top256
    "/BS/disentanglement/work/sae/imagenet_dinov2_vitl14_embeddings_val.npy:/BS/disentanglement/work/msae/8192_1024_TopKReLU_256_False_False_0.0_imagenet_dinov2_vitl14_embeddings_train.pth"
    "/BS/disentanglement/work/sae/imagenet_dinov2_vitl14_embeddings_val.npy:/BS/disentanglement/work/msae/16384_1024_TopKReLU_256_False_False_0.0_imagenet_dinov2_vitl14_embeddings_train.pth"
    "/BS/disentanglement/work/sae/imagenet_dinov2_vitl14_embeddings_val.npy:/BS/disentanglement/work/msae/32768_1024_TopKReLU_256_False_False_0.0_imagenet_dinov2_vitl14_embeddings_train.pth"
    # DINOv2 - BatchTop64
    "/BS/disentanglement/work/sae/imagenet_dinov2_vitl14_embeddings_val.npy:/BS/disentanglement/work/msae/batchtopk/8192_1024_TopKReLU_64_False_False_0.0_imagenet_dinov2_vitl14_embeddings_train.pth"
    "/BS/disentanglement/work/sae/imagenet_dinov2_vitl14_embeddings_val.npy:/BS/disentanglement/work/msae/batchtopk/16384_1024_TopKReLU_64_False_False_0.0_imagenet_dinov2_vitl14_embeddings_train.pth"
    "/BS/disentanglement/work/sae/imagenet_dinov2_vitl14_embeddings_val.npy:/BS/disentanglement/work/msae/batchtopk/32768_1024_TopKReLU_64_False_False_0.0_imagenet_dinov2_vitl14_embeddings_train.pth"
    # DINOv2 - BatchTop256
    "/BS/disentanglement/work/sae/imagenet_dinov2_vitl14_embeddings_val.npy:/BS/disentanglement/work/msae/batchtopk/8192_1024_TopKReLU_256_False_False_0.0_imagenet_dinov2_vitl14_embeddings_train.pth"
    "/BS/disentanglement/work/sae/imagenet_dinov2_vitl14_embeddings_val.npy:/BS/disentanglement/work/msae/batchtopk/16384_1024_TopKReLU_256_False_False_0.0_imagenet_dinov2_vitl14_embeddings_train.pth"
    "/BS/disentanglement/work/sae/imagenet_dinov2_vitl14_embeddings_val.npy:/BS/disentanglement/work/msae/batchtopk/32768_1024_TopKReLU_256_False_False_0.0_imagenet_dinov2_vitl14_embeddings_train.pth"
    # DINOv2 - MSAE RW
    "/BS/disentanglement/work/sae/imagenet_dinov2_vitl14_embeddings_val.npy:/BS/disentanglement/work/msae/8192_1024_TopKReLU_64_RW_False_False_0.0_imagenet_dinov2_vitl14_embeddings_train.pth"
    "/BS/disentanglement/work/sae/imagenet_dinov2_vitl14_embeddings_val.npy:/BS/disentanglement/work/msae/16384_1024_TopKReLU_64_RW_False_False_0.0_imagenet_dinov2_vitl14_embeddings_train.pth"
    "/BS/disentanglement/work/sae/imagenet_dinov2_vitl14_embeddings_val.npy:/BS/disentanglement/work/msae/32768_1024_TopKReLU_64_RW_False_False_0.0_imagenet_dinov2_vitl14_embeddings_train.pth"
    # DINOv2 - MSAE UW
    "/BS/disentanglement/work/sae/imagenet_dinov2_vitl14_embeddings_val.npy:/BS/disentanglement/work/msae/8192_1024_TopKReLU_64_UW_False_False_0.0_imagenet_dinov2_vitl14_embeddings_train.pth"
    "/BS/disentanglement/work/sae/imagenet_dinov2_vitl14_embeddings_val.npy:/BS/disentanglement/work/msae/16384_1024_TopKReLU_64_UW_False_False_0.0_imagenet_dinov2_vitl14_embeddings_train.pth"
    "/BS/disentanglement/work/sae/imagenet_dinov2_vitl14_embeddings_val.npy:/BS/disentanglement/work/msae/32768_1024_TopKReLU_64_UW_False_False_0.0_imagenet_dinov2_vitl14_embeddings_train.pth"
    # ViT-B/16 - ReLU
    "/BS/disentanglement/work/sae/imagenet_vit_b_16_embeddings_val.npy:/BS/disentanglement/work/msae/6144_768_ReLU_003_False_False_0.0_imagenet_vit_b_16_embeddings_train.pth"
    "/BS/disentanglement/work/sae/imagenet_vit_b_16_embeddings_val.npy:/BS/disentanglement/work/msae/12288_768_ReLU_003_False_False_0.0_imagenet_vit_b_16_embeddings_train.pth"
    "/BS/disentanglement/work/sae/imagenet_vit_b_16_embeddings_val.npy:/BS/disentanglement/work/msae/24576_768_ReLU_003_False_False_0.0_imagenet_vit_b_16_embeddings_train.pth"
    # ViT-B/16 - Top64
    "/BS/disentanglement/work/sae/imagenet_vit_b_16_embeddings_val.npy:/BS/disentanglement/work/msae/6144_768_TopKReLU_64_False_False_0.0_imagenet_vit_b_16_embeddings_train.pth"
    "/BS/disentanglement/work/sae/imagenet_vit_b_16_embeddings_val.npy:/BS/disentanglement/work/msae/12288_768_TopKReLU_64_False_False_0.0_imagenet_vit_b_16_embeddings_train.pth"
    "/BS/disentanglement/work/sae/imagenet_vit_b_16_embeddings_val.npy:/BS/disentanglement/work/msae/24576_768_TopKReLU_64_False_False_0.0_imagenet_vit_b_16_embeddings_train.pth"
    # ViT-B/16 - Top256
    "/BS/disentanglement/work/sae/imagenet_vit_b_16_embeddings_val.npy:/BS/disentanglement/work/msae/6144_768_TopKReLU_256_False_False_0.0_imagenet_vit_b_16_embeddings_train.pth"
    "/BS/disentanglement/work/sae/imagenet_vit_b_16_embeddings_val.npy:/BS/disentanglement/work/msae/12288_768_TopKReLU_256_False_False_0.0_imagenet_vit_b_16_embeddings_train.pth"
    "/BS/disentanglement/work/sae/imagenet_vit_b_16_embeddings_val.npy:/BS/disentanglement/work/msae/24576_768_TopKReLU_256_False_False_0.0_imagenet_vit_b_16_embeddings_train.pth"
    # ViT-B/16 - BatchTop64
    "/BS/disentanglement/work/sae/imagenet_vit_b_16_embeddings_val.npy:/BS/disentanglement/work/msae/batchtopk/6144_768_TopKReLU_64_False_False_0.0_imagenet_vit_b_16_embeddings_train.pth"
    "/BS/disentanglement/work/sae/imagenet_vit_b_16_embeddings_val.npy:/BS/disentanglement/work/msae/batchtopk/12288_768_TopKReLU_64_False_False_0.0_imagenet_vit_b_16_embeddings_train.pth"
    "/BS/disentanglement/work/sae/imagenet_vit_b_16_embeddings_val.npy:/BS/disentanglement/work/msae/batchtopk/24576_768_TopKReLU_64_False_False_0.0_imagenet_vit_b_16_embeddings_train.pth"
    # ViT-B/16 - BatchTop256
    "/BS/disentanglement/work/sae/imagenet_vit_b_16_embeddings_val.npy:/BS/disentanglement/work/msae/batchtopk/6144_768_TopKReLU_256_False_False_0.0_imagenet_vit_b_16_embeddings_train.pth"
    "/BS/disentanglement/work/sae/imagenet_vit_b_16_embeddings_val.npy:/BS/disentanglement/work/msae/batchtopk/12288_768_TopKReLU_256_False_False_0.0_imagenet_vit_b_16_embeddings_train.pth"
    "/BS/disentanglement/work/sae/imagenet_vit_b_16_embeddings_val.npy:/BS/disentanglement/work/msae/batchtopk/24576_768_TopKReLU_256_False_False_0.0_imagenet_vit_b_16_embeddings_train.pth"
    # ViT-B/16 - MSAE RW
    "/BS/disentanglement/work/sae/imagenet_vit_b_16_embeddings_val.npy:/BS/disentanglement/work/msae/6144_768_TopKReLU_64_RW_False_False_0.0_imagenet_vit_b_16_embeddings_train.pth"
    "/BS/disentanglement/work/sae/imagenet_vit_b_16_embeddings_val.npy:/BS/disentanglement/work/msae/12288_768_TopKReLU_64_RW_False_False_0.0_imagenet_vit_b_16_embeddings_train.pth"
    "/BS/disentanglement/work/sae/imagenet_vit_b_16_embeddings_val.npy:/BS/disentanglement/work/msae/24576_768_TopKReLU_64_RW_False_False_0.0_imagenet_vit_b_16_embeddings_train.pth"
    # ViT-B/16 - MSAE UW
    "/BS/disentanglement/work/sae/imagenet_vit_b_16_embeddings_val.npy:/BS/disentanglement/work/msae/6144_768_TopKReLU_64_UW_False_False_0.0_imagenet_vit_b_16_embeddings_train.pth"
    "/BS/disentanglement/work/sae/imagenet_vit_b_16_embeddings_val.npy:/BS/disentanglement/work/msae/12288_768_TopKReLU_64_UW_False_False_0.0_imagenet_vit_b_16_embeddings_train.pth"
    "/BS/disentanglement/work/sae/imagenet_vit_b_16_embeddings_val.npy:/BS/disentanglement/work/msae/24576_768_TopKReLU_64_UW_False_False_0.0_imagenet_vit_b_16_embeddings_train.pth"
    # ResNet50 - ReLU
    "/BS/disentanglement/work/sae/imagenet_resnet50_embeddings_val.npy:/BS/disentanglement/work/msae/16384_2048_ReLU_003_False_False_0.0_imagenet_resnet50_embeddings.pth"
    "/BS/disentanglement/work/sae/imagenet_resnet50_embeddings_val.npy:/BS/disentanglement/work/msae/32768_2048_ReLU_003_False_False_0.0_imagenet_resnet50_embeddings.pth"
    "/BS/disentanglement/work/sae/imagenet_resnet50_embeddings_val.npy:/BS/disentanglement/work/msae/65536_2048_ReLU_003_False_False_0.0_imagenet_resnet50_embeddings.pth"
    # ResNet50 - Top64
    "/BS/disentanglement/work/sae/imagenet_resnet50_embeddings_val.npy:/BS/disentanglement/work/msae/16384_2048_TopKReLU_64_False_False_0.0_imagenet_resnet50_embeddings.pth"
    "/BS/disentanglement/work/sae/imagenet_resnet50_embeddings_val.npy:/BS/disentanglement/work/msae/32768_2048_TopKReLU_64_False_False_0.0_imagenet_resnet50_embeddings.pth"
    "/BS/disentanglement/work/sae/imagenet_resnet50_embeddings_val.npy:/BS/disentanglement/work/msae/65536_2048_TopKReLU_64_False_False_0.0_imagenet_resnet50_embeddings.pth"
    # ResNet50 - Top256
    "/BS/disentanglement/work/sae/imagenet_resnet50_embeddings_val.npy:/BS/disentanglement/work/msae/16384_2048_TopKReLU_256_False_False_0.0_imagenet_resnet50_embeddings.pth"
    "/BS/disentanglement/work/sae/imagenet_resnet50_embeddings_val.npy:/BS/disentanglement/work/msae/32768_2048_TopKReLU_256_False_False_0.0_imagenet_resnet50_embeddings.pth"
    "/BS/disentanglement/work/sae/imagenet_resnet50_embeddings_val.npy:/BS/disentanglement/work/msae/65536_2048_TopKReLU_256_False_False_0.0_imagenet_resnet50_embeddings.pth"
    # ResNet50 - BatchTop64
    "/BS/disentanglement/work/sae/imagenet_resnet50_embeddings_val.npy:/BS/disentanglement/work/msae/batchtopk/16384_2048_TopKReLU_64_False_False_0.0_imagenet_resnet50_embeddings.pth"
    "/BS/disentanglement/work/sae/imagenet_resnet50_embeddings_val.npy:/BS/disentanglement/work/msae/batchtopk/32768_2048_TopKReLU_64_False_False_0.0_imagenet_resnet50_embeddings.pth"
    "/BS/disentanglement/work/sae/imagenet_resnet50_embeddings_val.npy:/BS/disentanglement/work/msae/batchtopk/65536_2048_TopKReLU_64_False_False_0.0_imagenet_resnet50_embeddings.pth"
    # ResNet50 - BatchTop256
    "/BS/disentanglement/work/sae/imagenet_resnet50_embeddings_val.npy:/BS/disentanglement/work/msae/batchtopk/16384_2048_TopKReLU_256_False_False_0.0_imagenet_resnet50_embeddings.pth"
    "/BS/disentanglement/work/sae/imagenet_resnet50_embeddings_val.npy:/BS/disentanglement/work/msae/batchtopk/32768_2048_TopKReLU_256_False_False_0.0_imagenet_resnet50_embeddings.pth"
    "/BS/disentanglement/work/sae/imagenet_resnet50_embeddings_val.npy:/BS/disentanglement/work/msae/batchtopk/65536_2048_TopKReLU_256_False_False_0.0_imagenet_resnet50_embeddings.pth"
    # ResNet50 - MSAE RW
    "/BS/disentanglement/work/sae/imagenet_resnet50_embeddings_val.npy:/BS/disentanglement/work/msae/16384_2048_TopKReLU_64_RW_False_False_0.0_imagenet_resnet50_embeddings.pth"
    "/BS/disentanglement/work/sae/imagenet_resnet50_embeddings_val.npy:/BS/disentanglement/work/msae/32768_2048_TopKReLU_64_RW_False_False_0.0_imagenet_resnet50_embeddings.pth"
    "/BS/disentanglement/work/sae/imagenet_resnet50_embeddings_val.npy:/BS/disentanglement/work/msae/65536_2048_TopKReLU_64_RW_False_False_0.0_imagenet_resnet50_embeddings.pth"
    # ResNet50 - MSAE UW
    "/BS/disentanglement/work/sae/imagenet_resnet50_embeddings_val.npy:/BS/disentanglement/work/msae/16384_2048_TopKReLU_64_UW_False_False_0.0_imagenet_resnet50_embeddings.pth"
    "/BS/disentanglement/work/sae/imagenet_resnet50_embeddings_val.npy:/BS/disentanglement/work/msae/32768_2048_TopKReLU_64_UW_False_False_0.0_imagenet_resnet50_embeddings.pth"
    "/BS/disentanglement/work/sae/imagenet_resnet50_embeddings_val.npy:/BS/disentanglement/work/msae/65536_2048_TopKReLU_64_UW_False_False_0.0_imagenet_resnet50_embeddings.pth"
)

# Get the configuration for this array task
CONFIG=${CONFIGS[$SLURM_ARRAY_TASK_ID]}

# Split the configuration into dataset and model
IFS=':' read -r DATASET MODEL <<< "$CONFIG"

echo "Processing task $SLURM_ARRAY_TASK_ID"
echo "Dataset: $DATASET"
echo "Model: $MODEL"

# Run the extraction
uv run /BS/disentanglement/work/Disentanglement/MSAE/extract_sae_embeddings.py \
    -m "$MODEL" \
    -d "$DATASET" \
    -o "$STORE_DIR"

echo "Task $SLURM_ARRAY_TASK_ID completed"