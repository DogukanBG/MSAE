#!/bin/bash
#SBATCH -p gpu16
#SBATCH --job-name=parallel_fms
#SBATCH -o /BS/disentanglement/work/Disentanglement/jobs/fms_parallel/fms-%a.out
#SBATCH --time=23:59:00
#SBATCH --gres gpu:1
#SBATCH --array=0-67%5

STORE_DIR="/BS/disentanglement/work/msae/extract_embeddings"

# Define all configurations in arrays
declare -a MODELS=()
declare -a SAES=()
declare -a EVAL_HC=()
declare -a TAUS=()

# Helper function to add configuration
add_config() {
    MODELS+=("$1")
    SAES+=("$2")
    EVAL_HC+=("${3:-False}")
    TAUS+=("${4:-0.0}")
}

# ViT-B/16 configurations
MODEL="vit_b_16"
# ReLU
add_config "$MODEL" "/BS/disentanglement/work/msae/6144_768_ReLU_003_False_False_0.0_imagenet_vit_b_16_embeddings_train.pth"
add_config "$MODEL" "/BS/disentanglement/work/msae/12288_768_ReLU_003_False_False_0.0_imagenet_vit_b_16_embeddings_train.pth"
add_config "$MODEL" "/BS/disentanglement/work/msae/24576_768_ReLU_003_False_False_0.0_imagenet_vit_b_16_embeddings_train.pth"
# Top256
add_config "$MODEL" "/BS/disentanglement/work/msae/6144_768_TopKReLU_256_False_False_0.0_imagenet_vit_b_16_embeddings_train.pth"
add_config "$MODEL" "/BS/disentanglement/work/msae/12288_768_TopKReLU_256_False_False_0.0_imagenet_vit_b_16_embeddings_train.pth"
add_config "$MODEL" "/BS/disentanglement/work/msae/24576_768_TopKReLU_256_False_False_0.0_imagenet_vit_b_16_embeddings_train.pth"
# Top64
add_config "$MODEL" "/BS/disentanglement/work/msae/6144_768_TopKReLU_64_False_False_0.0_imagenet_vit_b_16_embeddings_train.pth"
add_config "$MODEL" "/BS/disentanglement/work/msae/12288_768_TopKReLU_64_False_False_0.0_imagenet_vit_b_16_embeddings_train.pth"
add_config "$MODEL" "/BS/disentanglement/work/msae/24576_768_TopKReLU_64_False_False_0.0_imagenet_vit_b_16_embeddings_train.pth"
# BatchTop256
add_config "$MODEL" "/BS/disentanglement/work/msae/batchtopk/6144_768_TopKReLU_256_False_False_0.0_imagenet_vit_b_16_embeddings_train.pth"
add_config "$MODEL" "/BS/disentanglement/work/msae/batchtopk/12288_768_TopKReLU_256_False_False_0.0_imagenet_vit_b_16_embeddings_train.pth"
add_config "$MODEL" "/BS/disentanglement/work/msae/batchtopk/24576_768_TopKReLU_256_False_False_0.0_imagenet_vit_b_16_embeddings_train.pth"
# BatchTop64
add_config "$MODEL" "/BS/disentanglement/work/msae/batchtopk/6144_768_TopKReLU_64_False_False_0.0_imagenet_vit_b_16_embeddings_train.pth"
add_config "$MODEL" "/BS/disentanglement/work/msae/batchtopk/12288_768_TopKReLU_64_False_False_0.0_imagenet_vit_b_16_embeddings_train.pth"
add_config "$MODEL" "/BS/disentanglement/work/msae/batchtopk/24576_768_TopKReLU_64_False_False_0.0_imagenet_vit_b_16_embeddings_train.pth"
# MSAE RW
add_config "$MODEL" "/BS/disentanglement/work/msae/6144_768_TopKReLU_64_RW_False_False_0.0_imagenet_vit_b_16_embeddings_train.pth"
add_config "$MODEL" "/BS/disentanglement/work/msae/12288_768_TopKReLU_64_RW_False_False_0.0_imagenet_vit_b_16_embeddings_train.pth"
add_config "$MODEL" "/BS/disentanglement/work/msae/24576_768_TopKReLU_64_RW_False_False_0.0_imagenet_vit_b_16_embeddings_train.pth"
# MSAE UW
add_config "$MODEL" "/BS/disentanglement/work/msae/6144_768_TopKReLU_64_UW_False_False_0.0_imagenet_vit_b_16_embeddings_train.pth"
add_config "$MODEL" "/BS/disentanglement/work/msae/12288_768_TopKReLU_64_UW_False_False_0.0_imagenet_vit_b_16_embeddings_train.pth"
add_config "$MODEL" "/BS/disentanglement/work/msae/24576_768_TopKReLU_64_UW_False_False_0.0_imagenet_vit_b_16_embeddings_train.pth"

# ResNet50 configurations
MODEL="resnet50"
# ReLU
add_config "$MODEL" "/BS/disentanglement/work/msae/16384_2048_ReLU_003_False_False_0.0_imagenet_resnet50_embeddings.pth"
add_config "$MODEL" "/BS/disentanglement/work/msae/32768_2048_ReLU_003_False_False_0.0_imagenet_resnet50_embeddings.pth"
add_config "$MODEL" "/BS/disentanglement/work/msae/65536_2048_ReLU_003_False_False_0.0_imagenet_resnet50_embeddings.pth"
# Top256
add_config "$MODEL" "/BS/disentanglement/work/msae/16384_2048_TopKReLU_256_False_False_0.0_imagenet_resnet50_embeddings.pth"
add_config "$MODEL" "/BS/disentanglement/work/msae/32768_2048_TopKReLU_256_False_False_0.0_imagenet_resnet50_embeddings.pth"
add_config "$MODEL" "/BS/disentanglement/work/msae/65536_2048_TopKReLU_256_False_False_0.0_imagenet_resnet50_embeddings.pth"
# Top64
add_config "$MODEL" "/BS/disentanglement/work/msae/16384_2048_TopKReLU_64_False_False_0.0_imagenet_resnet50_embeddings.pth"
add_config "$MODEL" "/BS/disentanglement/work/msae/32768_2048_TopKReLU_64_False_False_0.0_imagenet_resnet50_embeddings.pth"
add_config "$MODEL" "/BS/disentanglement/work/msae/65536_2048_TopKReLU_64_False_False_0.0_imagenet_resnet50_embeddings.pth"
# BatchTop256
add_config "$MODEL" "/BS/disentanglement/work/msae/batchtopk/16384_2048_TopKReLU_256_False_False_0.0_imagenet_resnet50_embeddings.pth"
add_config "$MODEL" "/BS/disentanglement/work/msae/batchtopk/32768_2048_TopKReLU_256_False_False_0.0_imagenet_resnet50_embeddings.pth"
add_config "$MODEL" "/BS/disentanglement/work/msae/batchtopk/65536_2048_TopKReLU_256_False_False_0.0_imagenet_resnet50_embeddings.pth"
# BatchTop64
add_config "$MODEL" "/BS/disentanglement/work/msae/batchtopk/16384_2048_TopKReLU_64_False_False_0.0_imagenet_resnet50_embeddings.pth"
add_config "$MODEL" "/BS/disentanglement/work/msae/batchtopk/32768_2048_TopKReLU_64_False_False_0.0_imagenet_resnet50_embeddings.pth"
add_config "$MODEL" "/BS/disentanglement/work/msae/batchtopk/65536_2048_TopKReLU_64_False_False_0.0_imagenet_resnet50_embeddings.pth"
# MSAE RW
add_config "$MODEL" "/BS/disentanglement/work/msae/16384_2048_TopKReLU_64_RW_False_False_0.0_imagenet_resnet50_embeddings.pth"
add_config "$MODEL" "/BS/disentanglement/work/msae/32768_2048_TopKReLU_64_RW_False_False_0.0_imagenet_resnet50_embeddings.pth"
add_config "$MODEL" "/BS/disentanglement/work/msae/65536_2048_TopKReLU_64_RW_False_False_0.0_imagenet_resnet50_embeddings.pth"
# MSAE UW
add_config "$MODEL" "/BS/disentanglement/work/msae/16384_2048_TopKReLU_64_UW_False_False_0.0_imagenet_resnet50_embeddings.pth"
add_config "$MODEL" "/BS/disentanglement/work/msae/32768_2048_TopKReLU_64_UW_False_False_0.0_imagenet_resnet50_embeddings.pth"
add_config "$MODEL" "/BS/disentanglement/work/msae/65536_2048_TopKReLU_64_UW_False_False_0.0_imagenet_resnet50_embeddings.pth"

# DINOv2 configurations
MODEL="dinov2_vitl14"
# ReLU
add_config "$MODEL" "/BS/disentanglement/work/msae/8192_1024_ReLU_003_False_False_0.0_imagenet_dinov2_vitl14_embeddings_train.pth"
add_config "$MODEL" "/BS/disentanglement/work/msae/16384_1024_ReLU_003_False_False_0.0_imagenet_dinov2_vitl14_embeddings_train.pth"
add_config "$MODEL" "/BS/disentanglement/work/msae/32768_1024_ReLU_003_False_False_0.0_imagenet_dinov2_vitl14_embeddings_train.pth"
# Top256
add_config "$MODEL" "/BS/disentanglement/work/msae/8192_1024_TopKReLU_256_False_False_0.0_imagenet_dinov2_vitl14_embeddings_train.pth"
add_config "$MODEL" "/BS/disentanglement/work/msae/16384_1024_TopKReLU_256_False_False_0.0_imagenet_dinov2_vitl14_embeddings_train.pth"
add_config "$MODEL" "/BS/disentanglement/work/msae/32768_1024_TopKReLU_256_False_False_0.0_imagenet_dinov2_vitl14_embeddings_train.pth"
# Top64
add_config "$MODEL" "/BS/disentanglement/work/msae/8192_1024_TopKReLU_64_False_False_0.0_imagenet_dinov2_vitl14_embeddings_train.pth"
add_config "$MODEL" "/BS/disentanglement/work/msae/16384_1024_TopKReLU_64_False_False_0.0_imagenet_dinov2_vitl14_embeddings_train.pth"
add_config "$MODEL" "/BS/disentanglement/work/msae/32768_1024_TopKReLU_64_False_False_0.0_imagenet_dinov2_vitl14_embeddings_train.pth"
# BatchTop256
add_config "$MODEL" "/BS/disentanglement/work/msae/batchtopk/8192_1024_TopKReLU_256_False_False_0.0_imagenet_dinov2_vitl14_embeddings_train.pth"
add_config "$MODEL" "/BS/disentanglement/work/msae/batchtopk/16384_1024_TopKReLU_256_False_False_0.0_imagenet_dinov2_vitl14_embeddings_train.pth"
add_config "$MODEL" "/BS/disentanglement/work/msae/batchtopk/32768_1024_TopKReLU_256_False_False_0.0_imagenet_dinov2_vitl14_embeddings_train.pth"
# BatchTop64
add_config "$MODEL" "/BS/disentanglement/work/msae/batchtopk/8192_1024_TopKReLU_64_False_False_0.0_imagenet_dinov2_vitl14_embeddings_train.pth"
add_config "$MODEL" "/BS/disentanglement/work/msae/batchtopk/16384_1024_TopKReLU_64_False_False_0.0_imagenet_dinov2_vitl14_embeddings_train.pth"
add_config "$MODEL" "/BS/disentanglement/work/msae/batchtopk/32768_1024_TopKReLU_64_False_False_0.0_imagenet_dinov2_vitl14_embeddings_train.pth"
# MSAE RW
add_config "$MODEL" "/BS/disentanglement/work/msae/8192_1024_TopKReLU_64_RW_False_False_0.0_imagenet_dinov2_vitl14_embeddings_train.pth"
add_config "$MODEL" "/BS/disentanglement/work/msae/16384_1024_TopKReLU_64_RW_False_False_0.0_imagenet_dinov2_vitl14_embeddings_train.pth"
add_config "$MODEL" "/BS/disentanglement/work/msae/32768_1024_TopKReLU_64_RW_False_False_0.0_imagenet_dinov2_vitl14_embeddings_train.pth"
# MSAE UW
add_config "$MODEL" "/BS/disentanglement/work/msae/8192_1024_TopKReLU_64_UW_False_False_0.0_imagenet_dinov2_vitl14_embeddings_train.pth"
add_config "$MODEL" "/BS/disentanglement/work/msae/16384_1024_TopKReLU_64_UW_False_False_0.0_imagenet_dinov2_vitl14_embeddings_train.pth"
add_config "$MODEL" "/BS/disentanglement/work/msae/32768_1024_TopKReLU_64_UW_False_False_0.0_imagenet_dinov2_vitl14_embeddings_train.pth"

# HC evaluation configurations
add_config "vit_b_16" "" "True" "0.03"
add_config "vit_b_16" "" "True" "0.02"
add_config "vit_b_16" "" "True" "0.01"
add_config "resnet50" "" "True" "0.03"
add_config "resnet50" "" "True" "0.02"

# Get configuration for this array task
TASK_MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"
TASK_SAE="${SAES[$SLURM_ARRAY_TASK_ID]}"
TASK_EVAL_HC="${EVAL_HC[$SLURM_ARRAY_TASK_ID]}"
TASK_TAU="${TAUS[$SLURM_ARRAY_TASK_ID]}"

echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Model: $TASK_MODEL"
echo "SAE: $TASK_SAE"
echo "Eval HC: $TASK_EVAL_HC"
echo "Tau: $TASK_TAU"

# Run the appropriate command
if [ "$TASK_EVAL_HC" = "True" ]; then
    uv run /BS/disentanglement/work/Disentanglement/MSAE/fms.py -m "$TASK_MODEL" --eval-hc True --tau "$TASK_TAU"
else
    uv run /BS/disentanglement/work/Disentanglement/MSAE/fms.py -m "$TASK_MODEL" --sae "$TASK_SAE"
fi

echo "Task $SLURM_ARRAY_TASK_ID completed"