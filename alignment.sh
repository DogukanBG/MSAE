#!/bin/bash
#SBATCH -p gpu20
#SBATCH --job-name=alignment
#SBATCH -o /BS/disentanglement/work/Disentanglement/jobs/sae_training/alignment-%j.out
#SBATCH --time=12:00:00
#SBATCH --gres gpu:1

STORE_DIR="/BS/disentanglement/work/msae/extract_embeddings"

##ViTB16
# SAE="vit_b_16"
# SAE="/BS/disentanglement/work/msae/6144_768_ReLU_003_False_False_0.0_imagenet_vit_b_16_embeddings_train.pth"
# SAE2="/BS/disentanglement/work/msae2/6144_768_ReLU_003_False_False_0.0_imagenet_vit_b_16_embeddings_train.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/aligment_metric.py --sae_1 $SAE --sae_2 $SAE2

# SAE="/BS/disentanglement/work/msae/12288_768_ReLU_003_False_False_0.0_imagenet_vit_b_16_embeddings_train.pth"
# SAE2="/BS/disentanglement/work/msae2/12288_768_ReLU_003_False_False_0.0_imagenet_vit_b_16_embeddings_train.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/aligment_metric.py --sae_1 $SAE --sae_2 $SAE2

# SAE="/BS/disentanglement/work/msae/24576_768_ReLU_003_False_False_0.0_imagenet_vit_b_16_embeddings_train.pth"
# SAE2="/BS/disentanglement/work/msae2/24576_768_ReLU_003_False_False_0.0_imagenet_vit_b_16_embeddings_train.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/aligment_metric.py --sae_1 $SAE --sae_2 $SAE2

# SAE="/BS/disentanglement/work/msae/6144_768_TopKReLU_256_False_False_0.0_imagenet_vit_b_16_embeddings_train.pth"
# SAE2="/BS/disentanglement/work/msae2/6144_768_TopKReLU_256_False_False_0.0_imagenet_vit_b_16_embeddings_train.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/aligment_metric.py --sae_1 $SAE --sae_2 $SAE2

# SAE="/BS/disentanglement/work/msae/12288_768_TopKReLU_256_False_False_0.0_imagenet_vit_b_16_embeddings_train.pth"
# SAE2="/BS/disentanglement/work/msae2/12288_768_TopKReLU_256_False_False_0.0_imagenet_vit_b_16_embeddings_train.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/aligment_metric.py --sae_1 $SAE --sae_2 $SAE2

# SAE="/BS/disentanglement/work/msae/24576_768_TopKReLU_256_False_False_0.0_imagenet_vit_b_16_embeddings_train.pth"
# SAE2="/BS/disentanglement/work/msae2/24576_768_TopKReLU_256_False_False_0.0_imagenet_vit_b_16_embeddings_train.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/aligment_metric.py --sae_1 $SAE --sae_2 $SAE2

# SAE="/BS/disentanglement/work/msae/6144_768_TopKReLU_64_False_False_0.0_imagenet_vit_b_16_embeddings_train.pth"
# SAE2="/BS/disentanglement/work/msae2/6144_768_TopKReLU_64_False_False_0.0_imagenet_vit_b_16_embeddings_train.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/aligment_metric.py --sae_1 $SAE --sae_2 $SAE2

# SAE="/BS/disentanglement/work/msae/12288_768_TopKReLU_64_False_False_0.0_imagenet_vit_b_16_embeddings_train.pth"
# SAE2="/BS/disentanglement/work/msae2/12288_768_TopKReLU_64_False_False_0.0_imagenet_vit_b_16_embeddings_train.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/aligment_metric.py --sae_1 $SAE --sae_2 $SAE2

# SAE="/BS/disentanglement/work/msae/24576_768_TopKReLU_64_False_False_0.0_imagenet_vit_b_16_embeddings_train.pth"
# SAE2="/BS/disentanglement/work/msae2/24576_768_TopKReLU_64_False_False_0.0_imagenet_vit_b_16_embeddings_train.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/aligment_metric.py --sae_1 $SAE --sae_2 $SAE2

# echo "ViT-b/16 evaluated"

# ##ResNet50
# SAE="resnet50"
# SAE="/BS/disentanglement/work/msae/16384_2048_ReLU_003_False_False_0.0_imagenet_resnet50_embeddings.pth"
# SAE2="/BS/disentanglement/work/msae2/16384_2048_ReLU_003_False_False_0.0_imagenet_resnet50_embeddings.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/aligment_metric.py --sae_1 $SAE --sae_2 $SAE2

# SAE="/BS/disentanglement/work/msae/32768_2048_ReLU_003_False_False_0.0_imagenet_resnet50_embeddings.pth"
# SAE2="/BS/disentanglement/work/msae2/32768_2048_ReLU_003_False_False_0.0_imagenet_resnet50_embeddings.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/aligment_metric.py --sae_1 $SAE --sae_2 $SAE2

# SAE="/BS/disentanglement/work/msae/65536_2048_ReLU_003_False_False_0.0_imagenet_resnet50_embeddings.pth"
# SAE2="/BS/disentanglement/work/msae2/65536_2048_ReLU_003_False_False_0.0_imagenet_resnet50_embeddings.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/aligment_metric.py --sae_1 $SAE --sae_2 $SAE2

# SAE="/BS/disentanglement/work/msae/16384_2048_TopKReLU_256_False_False_0.0_imagenet_resnet50_embeddings.pth"
# SAE2="/BS/disentanglement/work/msae2/16384_2048_TopKReLU_256_False_False_0.0_imagenet_resnet50_embeddings.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/aligment_metric.py --sae_1 $SAE --sae_2 $SAE2

# SAE="/BS/disentanglement/work/msae/32768_2048_TopKReLU_256_False_False_0.0_imagenet_resnet50_embeddings.pth"
# SAE2="/BS/disentanglement/work/msae2/32768_2048_TopKReLU_256_False_False_0.0_imagenet_resnet50_embeddings.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/aligment_metric.py --sae_1 $SAE --sae_2 $SAE2

# SAE="/BS/disentanglement/work/msae/65536_2048_TopKReLU_256_False_False_0.0_imagenet_resnet50_embeddings.pth"
# SAE2="/BS/disentanglement/work/msae2/65536_2048_TopKReLU_256_False_False_0.0_imagenet_resnet50_embeddings.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/aligment_metric.py --sae_1 $SAE --sae_2 $SAE2

# SAE="/BS/disentanglement/work/msae/16384_2048_TopKReLU_64_False_False_0.0_imagenet_resnet50_embeddings.pth"
# SAE2="/BS/disentanglement/work/msae2/16384_2048_TopKReLU_64_False_False_0.0_imagenet_resnet50_embeddings.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/aligment_metric.py --sae_1 $SAE --sae_2 $SAE2

# SAE="/BS/disentanglement/work/msae/32768_2048_TopKReLU_64_False_False_0.0_imagenet_resnet50_embeddings.pth"
# SAE2="/BS/disentanglement/work/msae2/32768_2048_TopKReLU_64_False_False_0.0_imagenet_resnet50_embeddings.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/aligment_metric.py --sae_1 $SAE --sae_2 $SAE2

# SAE="/BS/disentanglement/work/msae/65536_2048_TopKReLU_64_False_False_0.0_imagenet_resnet50_embeddings.pth"
# SAE2="/BS/disentanglement/work/msae2/65536_2048_TopKReLU_64_False_False_0.0_imagenet_resnet50_embeddings.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/aligment_metric.py --sae_1 $SAE --sae_2 $SAE2

# echo "ResNet50 SAE evaluated"

## DINOv2
# SAE="dinov2_vitl14"
# SAE="/BS/disentanglement/work/msae/8192_1024_ReLU_003_False_False_0.0_imagenet_dinov2_vitl14_embeddings_train.pth"
# SAE2="/BS/disentanglement/work/msae2/8192_1024_ReLU_003_False_False_0.0_imagenet_dinov2_vitl14_embeddings_train.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/aligment_metric.py --sae_1 $SAE --sae_2 $SAE2

# SAE="/BS/disentanglement/work/msae/16384_1024_ReLU_003_False_False_0.0_imagenet_dinov2_vitl14_embeddings_train.pth"
# SAE2="/BS/disentanglement/work/msae2/16384_1024_ReLU_003_False_False_0.0_imagenet_dinov2_vitl14_embeddings_train.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/aligment_metric.py --sae_1 $SAE --sae_2 $SAE2

# SAE="/BS/disentanglement/work/msae/32768_1024_ReLU_003_False_False_0.0_imagenet_dinov2_vitl14_embeddings_train.pth"
# SAE2="/BS/disentanglement/work/msae2/32768_1024_ReLU_003_False_False_0.0_imagenet_dinov2_vitl14_embeddings_train.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/aligment_metric.py --sae_1 $SAE --sae_2 $SAE2

# SAE="/BS/disentanglement/work/msae/8192_1024_TopKReLU_256_False_False_0.0_imagenet_dinov2_vitl14_embeddings_train.pth"
# SAE2="/BS/disentanglement/work/msae2/8192_1024_TopKReLU_256_False_False_0.0_imagenet_dinov2_vitl14_embeddings_train.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/aligment_metric.py --sae_1 $SAE --sae_2 $SAE2

# SAE="/BS/disentanglement/work/msae/16384_1024_TopKReLU_256_False_False_0.0_imagenet_dinov2_vitl14_embeddings_train.pth"
# SAE2="/BS/disentanglement/work/msae2/16384_1024_TopKReLU_256_False_False_0.0_imagenet_dinov2_vitl14_embeddings_train.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/aligment_metric.py --sae_1 $SAE --sae_2 $SAE2

# SAE="/BS/disentanglement/work/msae/32768_1024_TopKReLU_256_False_False_0.0_imagenet_dinov2_vitl14_embeddings_train.pth"
# SAE2="/BS/disentanglement/work/msae2/32768_1024_TopKReLU_256_False_False_0.0_imagenet_dinov2_vitl14_embeddings_train.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/aligment_metric.py --sae_1 $SAE --sae_2 $SAE2


# SAE="/BS/disentanglement/work/msae/8192_1024_TopKReLU_64_False_False_0.0_imagenet_dinov2_vitl14_embeddings_train.pth"
# SAE2="/BS/disentanglement/work/msae2/8192_1024_TopKReLU_64_False_False_0.0_imagenet_dinov2_vitl14_embeddings_train.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/aligment_metric.py --sae_1 $SAE --sae_2 $SAE2

# SAE="/BS/disentanglement/work/msae/16384_1024_TopKReLU_64_False_False_0.0_imagenet_dinov2_vitl14_embeddings_train.pth"
# SAE2="/BS/disentanglement/work/msae2/16384_1024_TopKReLU_64_False_False_0.0_imagenet_dinov2_vitl14_embeddings_train.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/aligment_metric.py --sae_1 $SAE --sae_2 $SAE2

# SAE="/BS/disentanglement/work/msae/32768_1024_TopKReLU_64_False_False_0.0_imagenet_dinov2_vitl14_embeddings_train.pth"
# SAE2="/BS/disentanglement/work/msae2/32768_1024_TopKReLU_64_False_False_0.0_imagenet_dinov2_vitl14_embeddings_train.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/aligment_metric.py --sae_1 $SAE --sae_2 $SAE2


# #BatchTop64
# SAE="/BS/disentanglement/work/msae/batchtopk/16384_1024_TopKReLU_64_False_False_0.0_imagenet_dinov2_vitl14_embeddings_train.pth"
# SAE2="/BS/disentanglement/work/msae2/batchtopk/16384_1024_TopKReLU_64_False_False_0.0_imagenet_dinov2_vitl14_embeddings_train.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/aligment_metric.py --sae_1 $SAE --sae_2 $SAE2


# SAE="/BS/disentanglement/work/msae/batchtopk/32768_1024_TopKReLU_64_False_False_0.0_imagenet_dinov2_vitl14_embeddings_train.pth"
# SAE2="/BS/disentanglement/work/msae2/batchtopk/32768_1024_TopKReLU_64_False_False_0.0_imagenet_dinov2_vitl14_embeddings_train.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/aligment_metric.py --sae_1 $SAE --sae_2 $SAE2


# SAE="/BS/disentanglement/work/msae/batchtopk/8192_1024_TopKReLU_64_False_False_0.0_imagenet_dinov2_vitl14_embeddings_train.pth"
# SAE2="/BS/disentanglement/work/msae2/batchtopk/8192_1024_TopKReLU_64_False_False_0.0_imagenet_dinov2_vitl14_embeddings_train.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/aligment_metric.py --sae_1 $SAE --sae_2 $SAE2


# #BatchTop256
# SAE="/BS/disentanglement/work/msae/batchtopk/16384_1024_TopKReLU_256_False_False_0.0_imagenet_dinov2_vitl14_embeddings_train.pth"
# SAE2="/BS/disentanglement/work/msae2/batchtopk/16384_1024_TopKReLU_256_False_False_0.0_imagenet_dinov2_vitl14_embeddings_train.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/aligment_metric.py --sae_1 $SAE --sae_2 $SAE2


# SAE="/BS/disentanglement/work/msae/batchtopk/32768_1024_TopKReLU_256_False_False_0.0_imagenet_dinov2_vitl14_embeddings_train.pth"
# SAE2="/BS/disentanglement/work/msae2/batchtopk/32768_1024_TopKReLU_256_False_False_0.0_imagenet_dinov2_vitl14_embeddings_train.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/aligment_metric.py --sae_1 $SAE --sae_2 $SAE2


# SAE="/BS/disentanglement/work/msae/batchtopk/8192_1024_TopKReLU_256_False_False_0.0_imagenet_dinov2_vitl14_embeddings_train.pth"
# SAE2="/BS/disentanglement/work/msae2/batchtopk/8192_1024_TopKReLU_256_False_False_0.0_imagenet_dinov2_vitl14_embeddings_train.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/aligment_metric.py --sae_1 $SAE --sae_2 $SAE2


# #MSAE RW
# SAE="/BS/disentanglement/work/msae/16384_1024_TopKReLU_64_RW_False_False_0.0_imagenet_dinov2_vitl14_embeddings_train.pth"
# SAE2="/BS/disentanglement/work/msae2/16384_1024_TopKReLU_64_RW_False_False_0.0_imagenet_dinov2_vitl14_embeddings_train.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/aligment_metric.py --sae_1 $SAE --sae_2 $SAE2


# SAE="/BS/disentanglement/work/msae/32768_1024_TopKReLU_64_RW_False_False_0.0_imagenet_dinov2_vitl14_embeddings_train.pth"
# SAE2="/BS/disentanglement/work/msae2/32768_1024_TopKReLU_64_RW_False_False_0.0_imagenet_dinov2_vitl14_embeddings_train.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/aligment_metric.py --sae_1 $SAE --sae_2 $SAE2


# SAE="/BS/disentanglement/work/msae/8192_1024_TopKReLU_64_RW_False_False_0.0_imagenet_dinov2_vitl14_embeddings_train.pth"
# SAE2="/BS/disentanglement/work/msae2/8192_1024_TopKReLU_64_RW_False_False_0.0_imagenet_dinov2_vitl14_embeddings_train.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/aligment_metric.py --sae_1 $SAE --sae_2 $SAE2


# #MSAE UW
SAE="/BS/disentanglement/work/msae/16384_1024_TopKReLU_64_UW_False_False_0.0_imagenet_dinov2_vitl14_embeddings_train.pth"
SAE2="/BS/disentanglement/work/msae2/16384_1024_TopKReLU_64_UW_False_False_0.0_imagenet_dinov2_vitl14_embeddings_train.pth"
uv run /BS/disentanglement/work/Disentanglement/MSAE/aligment_metric.py --sae_1 $SAE --sae_2 $SAE2


# SAE="/BS/disentanglement/work/msae/32768_1024_TopKReLU_64_UW_False_False_0.0_imagenet_dinov2_vitl14_embeddings_train.pth"
# SAE2="/BS/disentanglement/work/msae2/32768_1024_TopKReLU_64_UW_False_False_0.0_imagenet_dinov2_vitl14_embeddings_train.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/aligment_metric.py --sae_1 $SAE --sae_2 $SAE2


# SAE="/BS/disentanglement/work/msae/8192_1024_TopKReLU_64_UW_False_False_0.0_imagenet_dinov2_vitl14_embeddings_train.pth"
# SAE2="/BS/disentanglement/work/msae2/8192_1024_TopKReLU_64_UW_False_False_0.0_imagenet_dinov2_vitl14_embeddings_train.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/aligment_metric.py --sae_1 $SAE --sae_2 $SAE2




echo "DINOv2 evaluated"