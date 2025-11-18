#!/bin/bash
#SBATCH -p gpu22
#SBATCH -t 23:59:00
#SBATCH -o /BS/disentanglement/work/Disentanglement/experiments/dinov2_base_eval/dinov2b-rest2.out
#SBATCH --gres gpu:1

export WANDB_API_KEY="a6f51864caf2d973f427aef4ccc067fd64c91bcd"

# Define all model-dataset combinations
declare -a CONFIGS=(
    
    # DINOv2 - ReLU
    #"/BS/disentanglement/work/sae/imagenet_dinov2_vitb14_embeddings_val.npy:/BS/disentanglement/work/msae/768_768_ReLU_003_False_False_0.0_imagenet_dinov2_vitb14_embeddings_train.pth"
    #"/BS/disentanglement/work/sae/imagenet_dinov2_vitb14_embeddings_val.npy:/BS/disentanglement/work/msae/1536_768_ReLU_003_False_False_0.0_imagenet_dinov2_vitb14_embeddings_train.pth"
    #"/BS/disentanglement/work/sae/imagenet_dinov2_vitb14_embeddings_val.npy:/BS/disentanglement/work/msae/3072_768_ReLU_003_False_False_0.0_imagenet_dinov2_vitb14_embeddings_train.pth"
    
    #"/BS/disentanglement/work/sae/imagenet_dinov2_vitb14_embeddings_val.npy:/BS/disentanglement/work/msae/6144_768_ReLU_003_False_False_0.0_imagenet_dinov2_vitb14_embeddings_train.pth"
    #"/BS/disentanglement/work/sae/imagenet_dinov2_vitb14_embeddings_val.npy:/BS/disentanglement/work/msae/12288_768_ReLU_003_False_False_0.0_imagenet_dinov2_vitb14_embeddings_train.pth"
    #"/BS/disentanglement/work/sae/imagenet_dinov2_vitb14_embeddings_val.npy:/BS/disentanglement/work/msae/24576_768_ReLU_003_False_False_0.0_imagenet_dinov2_vitb14_embeddings_train.pth"
    
    # # DINOv2 - Top64
    #"/BS/disentanglement/work/sae/imagenet_dinov2_vitb14_embeddings_val.npy:/BS/disentanglement/work/msae/768_768_TopKReLU_64_False_False_0.0_imagenet_dinov2_vitb14_embeddings_train.pth"
    #"/BS/disentanglement/work/sae/imagenet_dinov2_vitb14_embeddings_val.npy:/BS/disentanglement/work/msae/1536_768_TopKReLU_64_False_False_0.0_imagenet_dinov2_vitb14_embeddings_train.pth"
    #"/BS/disentanglement/work/sae/imagenet_dinov2_vitb14_embeddings_val.npy:/BS/disentanglement/work/msae/3072_768_TopKReLU_64_False_False_0.0_imagenet_dinov2_vitb14_embeddings_train.pth"

    #"/BS/disentanglement/work/sae/imagenet_dinov2_vitb14_embeddings_val.npy:/BS/disentanglement/work/msae/6144_768_TopKReLU_64_False_False_0.0_imagenet_dinov2_vitb14_embeddings_train.pth"
    #"/BS/disentanglement/work/sae/imagenet_dinov2_vitb14_embeddings_val.npy:/BS/disentanglement/work/msae/12288_768_TopKReLU_64_False_False_0.0_imagenet_dinov2_vitb14_embeddings_train.pth"
    #"/BS/disentanglement/work/sae/imagenet_dinov2_vitb14_embeddings_val.npy:/BS/disentanglement/work/msae/24576_768_TopKReLU_64_False_False_0.0_imagenet_dinov2_vitb14_embeddings_train.pth"
    
    # # DINOv2 - Top256
    #"/BS/disentanglement/work/sae/imagenet_dinov2_vitb14_embeddings_val.npy:/BS/disentanglement/work/msae/768_768_TopKReLU_256_False_False_0.0_imagenet_dinov2_vitb14_embeddings_train.pth"
    #"/BS/disentanglement/work/sae/imagenet_dinov2_vitb14_embeddings_val.npy:/BS/disentanglement/work/msae/1536_768_TopKReLU_256_False_False_0.0_imagenet_dinov2_vitb14_embeddings_train.pth"
    #"/BS/disentanglement/work/sae/imagenet_dinov2_vitb14_embeddings_val.npy:/BS/disentanglement/work/msae/3072_768_TopKReLU_256_False_False_0.0_imagenet_dinov2_vitb14_embeddings_train.pth"

    #"/BS/disentanglement/work/sae/imagenet_dinov2_vitb14_embeddings_val.npy:/BS/disentanglement/work/msae/6144_768_TopKReLU_256_False_False_0.0_imagenet_dinov2_vitb14_embeddings_train.pth"
    #"/BS/disentanglement/work/sae/imagenet_dinov2_vitb14_embeddings_val.npy:/BS/disentanglement/work/msae/12288_768_TopKReLU_256_False_False_0.0_imagenet_dinov2_vitb14_embeddings_train.pth"
    #"/BS/disentanglement/work/sae/imagenet_dinov2_vitb14_embeddings_val.npy:/BS/disentanglement/work/msae/24576_768_TopKReLU_256_False_False_0.0_imagenet_dinov2_vitb14_embeddings_train.pth"
    
    # # DINOv2 - BatchTop64
    #"/BS/disentanglement/work/sae/imagenet_dinov2_vitb14_embeddings_val.npy:/BS/disentanglement/work/msae/batchtopk/768_768_TopKReLU_64_False_False_0.0_imagenet_dinov2_vitb14_embeddings_train.pth"
    #"/BS/disentanglement/work/sae/imagenet_dinov2_vitb14_embeddings_val.npy:/BS/disentanglement/work/msae/batchtopk/1536_768_TopKReLU_64_False_False_0.0_imagenet_dinov2_vitb14_embeddings_train.pth"
    #"/BS/disentanglement/work/sae/imagenet_dinov2_vitb14_embeddings_val.npy:/BS/disentanglement/work/msae/batchtopk/3072_768_TopKReLU_64_False_False_0.0_imagenet_dinov2_vitb14_embeddings_train.pth"

    #"/BS/disentanglement/work/sae/imagenet_dinov2_vitb14_embeddings_val.npy:/BS/disentanglement/work/msae/batchtopk/6144_768_TopKReLU_64_False_False_0.0_imagenet_dinov2_vitb14_embeddings_train.pth"
    #"/BS/disentanglement/work/sae/imagenet_dinov2_vitb14_embeddings_val.npy:/BS/disentanglement/work/msae/batchtopk/12288_768_TopKReLU_64_False_False_0.0_imagenet_dinov2_vitb14_embeddings_train.pth"
    #"/BS/disentanglement/work/sae/imagenet_dinov2_vitb14_embeddings_val.npy:/BS/disentanglement/work/msae/batchtopk/24576_768_TopKReLU_64_False_False_0.0_imagenet_dinov2_vitb14_embeddings_train.pth"
    
    # # DINOv2 - BatchTop256
    
        #"/BS/disentanglement/work/sae/imagenet_dinov2_vitb14_embeddings_val.npy:/BS/disentanglement/work/msae/batchtopk/768_768_TopKReLU_256_False_False_0.0_imagenet_dinov2_vitb14_embeddings_train.pth"
        #"/BS/disentanglement/work/sae/imagenet_dinov2_vitb14_embeddings_val.npy:/BS/disentanglement/work/msae/batchtopk/1536_768_TopKReLU_256_False_False_0.0_imagenet_dinov2_vitb14_embeddings_train.pth"
        #"/BS/disentanglement/work/sae/imagenet_dinov2_vitb14_embeddings_val.npy:/BS/disentanglement/work/msae/batchtopk/3072_768_TopKReLU_256_False_False_0.0_imagenet_dinov2_vitb14_embeddings_train.pth"

        "/BS/disentanglement/work/sae/imagenet_dinov2_vitb14_embeddings_val.npy:/BS/disentanglement/work/msae/batchtopk/6144_768_TopKReLU_256_False_False_0.0_imagenet_dinov2_vitb14_embeddings_train.pth"
        "/BS/disentanglement/work/sae/imagenet_dinov2_vitb14_embeddings_val.npy:/BS/disentanglement/work/msae/batchtopk/12288_768_TopKReLU_256_False_False_0.0_imagenet_dinov2_vitb14_embeddings_train.pth"
        "/BS/disentanglement/work/sae/imagenet_dinov2_vitb14_embeddings_val.npy:/BS/disentanglement/work/msae/batchtopk/24576_768_TopKReLU_256_False_False_0.0_imagenet_dinov2_vitb14_embeddings_train.pth"
    
    #DINOv2 - MSAE RW
    #"/BS/disentanglement/work/sae/imagenet_dinov2_vitb14_embeddings_val.npy:/BS/disentanglement/work/msae/768_768_TopKReLU_64_RW_False_False_0.0_imagenet_dinov2_vitb14_embeddings_train.pth"
    #"/BS/disentanglement/work/sae/imagenet_dinov2_vitb14_embeddings_val.npy:/BS/disentanglement/work/msae/1536_768_TopKReLU_64_RW_False_False_0.0_imagenet_dinov2_vitb14_embeddings_train.pth"
    #"/BS/disentanglement/work/sae/imagenet_dinov2_vitb14_embeddings_val.npy:/BS/disentanglement/work/msae/3072_768_TopKReLU_64_RW_False_False_0.0_imagenet_dinov2_vitb14_embeddings_train.pth"

    #"/BS/disentanglement/work/sae/imagenet_dinov2_vitb14_embeddings_val.npy:/BS/disentanglement/work/msae/6144_768_TopKReLU_64_RW_False_False_0.0_imagenet_dinov2_vitb14_embeddings_train.pth"
    #"/BS/disentanglement/work/sae/imagenet_dinov2_vitb14_embeddings_val.npy:/BS/disentanglement/work/msae/12288_768_TopKReLU_64_RW_False_False_0.0_imagenet_dinov2_vitb14_embeddings_train.pth"
    #"/BS/disentanglement/work/sae/imagenet_dinov2_vitb14_embeddings_val.npy:/BS/disentanglement/work/msae/24576_768_TopKReLU_64_RW_False_False_0.0_imagenet_dinov2_vitb14_embeddings_train.pth"
    
    # # DINOv2 - MSAE UW
    #"/BS/disentanglement/work/sae/imagenet_dinov2_vitb14_embeddings_val.npy:/BS/disentanglement/work/msae/768_768_TopKReLU_64_UW_False_False_0.0_imagenet_dinov2_vitb14_embeddings_train.pth"
    #"/BS/disentanglement/work/sae/imagenet_dinov2_vitb14_embeddings_val.npy:/BS/disentanglement/work/msae/1536_768_TopKReLU_64_UW_False_False_0.0_imagenet_dinov2_vitb14_embeddings_train.pth"
    #"/BS/disentanglement/work/sae/imagenet_dinov2_vitb14_embeddings_val.npy:/BS/disentanglement/work/msae/3072_768_TopKReLU_64_UW_False_False_0.0_imagenet_dinov2_vitb14_embeddings_train.pth"

    #"/BS/disentanglement/work/sae/imagenet_dinov2_vitb14_embeddings_val.npy:/BS/disentanglement/work/msae/6144_768_TopKReLU_64_UW_False_False_0.0_imagenet_dinov2_vitb14_embeddings_train.pth"
    #"/BS/disentanglement/work/sae/imagenet_dinov2_vitb14_embeddings_val.npy:/BS/disentanglement/work/msae/12288_768_TopKReLU_64_UW_False_False_0.0_imagenet_dinov2_vitb14_embeddings_train.pth"
    #"/BS/disentanglement/work/sae/imagenet_dinov2_vitb14_embeddings_val.npy:/BS/disentanglement/work/msae/24576_768_TopKReLU_64_UW_False_False_0.0_imagenet_dinov2_vitb14_embeddings_train.pth"
    
)

# Run all configurations sequentially
for i in "${!CONFIGS[@]}"; do
    CONFIG="${CONFIGS[$i]}"
    
    # Split the configuration into dataset and model
    IFS=':' read -r DATASET MODEL <<< "$CONFIG"
    
    echo "Processing task $i of ${#CONFIGS[@]}"
    echo "Dataset: $DATASET"
    echo "Model: $MODEL"
    
    # Run the extraction
    uv run /BS/disentanglement/work/Disentanglement/MSAE/extract_sae_embeddings.py \
        -m "$MODEL" \
        -d "$DATASET" \
        -o "$STORE_DIR" \
        --compute-monosemanticity
    
    echo "Task $i completed"
    echo "---"
done

echo "All tasks completed"