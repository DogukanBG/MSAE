#!/bin/bash
#SBATCH -p gpu22
#SBATCH --job-name=fms_sae
#SBATCH -o /BS/disentanglement/work/Disentanglement/jobs/sae_training/fms-%j.out
#SBATCH --time=23:59:00
#SBATCH --gres gpu:1

STORE_DIR="/BS/disentanglement/work/msae/extract_embeddings"

##ViTB16
# MODEL="vit_b_16"
# #ReLU
# SAE="/BS/disentanglement/work/msae/6144_768_ReLU_003_False_False_0.0_imagenet_vit_b_16_embeddings_train.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/fms.py -m $MODEL --sae $SAE  

# SAE="/BS/disentanglement/work/msae/12288_768_ReLU_003_False_False_0.0_imagenet_vit_b_16_embeddings_train.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/fms.py -m $MODEL --sae $SAE  

# SAE="/BS/disentanglement/work/msae/24576_768_ReLU_003_False_False_0.0_imagenet_vit_b_16_embeddings_train.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/fms.py -m $MODEL --sae $SAE  

# #Top256
# SAE="/BS/disentanglement/work/msae/6144_768_TopKReLU_256_False_False_0.0_imagenet_vit_b_16_embeddings_train.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/fms.py -m $MODEL --sae $SAE  

# SAE="/BS/disentanglement/work/msae/12288_768_TopKReLU_256_False_False_0.0_imagenet_vit_b_16_embeddings_train.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/fms.py -m $MODEL --sae $SAE  

# SAE="/BS/disentanglement/work/msae/24576_768_TopKReLU_256_False_False_0.0_imagenet_vit_b_16_embeddings_train.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/fms.py -m $MODEL --sae $SAE  

# #Top64
# SAE="/BS/disentanglement/work/msae/6144_768_TopKReLU_64_False_False_0.0_imagenet_vit_b_16_embeddings_train.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/fms.py -m $MODEL --sae $SAE  

# SAE="/BS/disentanglement/work/msae/12288_768_TopKReLU_64_False_False_0.0_imagenet_vit_b_16_embeddings_train.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/fms.py -m $MODEL --sae $SAE  

# SAE="/BS/disentanglement/work/msae/24576_768_TopKReLU_64_False_False_0.0_imagenet_vit_b_16_embeddings_train.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/fms.py -m $MODEL --sae $SAE  

# #BatchTop256
# SAE="/BS/disentanglement/work/msae/batchtopk/6144_768_TopKReLU_256_False_False_0.0_imagenet_vit_b_16_embeddings_train.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/fms.py -m $MODEL --sae $SAE  

# SAE="/BS/disentanglement/work/msae/batchtopk/12288_768_TopKReLU_256_False_False_0.0_imagenet_vit_b_16_embeddings_train.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/fms.py -m $MODEL --sae $SAE  

# SAE="/BS/disentanglement/work/msae/batchtopk/24576_768_TopKReLU_256_False_False_0.0_imagenet_vit_b_16_embeddings_train.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/fms.py -m $MODEL --sae $SAE  

# #BatchTop64
# SAE="/BS/disentanglement/work/msae/batchtopk/6144_768_TopKReLU_64_False_False_0.0_imagenet_vit_b_16_embeddings_train.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/fms.py -m $MODEL --sae $SAE  

# SAE="/BS/disentanglement/work/msae/batchtopk/12288_768_TopKReLU_64_False_False_0.0_imagenet_vit_b_16_embeddings_train.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/fms.py -m $MODEL --sae $SAE  

# SAE="/BS/disentanglement/work/msae/batchtopk/24576_768_TopKReLU_64_False_False_0.0_imagenet_vit_b_16_embeddings_train.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/fms.py -m $MODEL --sae $SAE  

# #MSAE RW
# SAE="/BS/disentanglement/work/msae/6144_768_TopKReLU_64_RW_False_False_0.0_imagenet_vit_b_16_embeddings_train.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/fms.py -m $MODEL --sae $SAE  

# SAE="/BS/disentanglement/work/msae/12288_768_TopKReLU_64_RW_False_False_0.0_imagenet_vit_b_16_embeddings_train.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/fms.py -m $MODEL --sae $SAE  

# SAE="/BS/disentanglement/work/msae/24576_768_TopKReLU_64_RW_False_False_0.0_imagenet_vit_b_16_embeddings_train.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/fms.py -m $MODEL --sae $SAE  

# #MSAE UW
# SAE="/BS/disentanglement/work/msae/6144_768_TopKReLU_64_UW_False_False_0.0_imagenet_vit_b_16_embeddings_train.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/fms.py -m $MODEL --sae $SAE  

# SAE="/BS/disentanglement/work/msae/12288_768_TopKReLU_64_UW_False_False_0.0_imagenet_vit_b_16_embeddings_train.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/fms.py -m $MODEL --sae $SAE  

# SAE="/BS/disentanglement/work/msae/24576_768_TopKReLU_64_UW_False_False_0.0_imagenet_vit_b_16_embeddings_train.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/fms.py -m $MODEL --sae $SAE  

# echo "ViT-b/16 evaluated"

# ##ResNet50
# MODEL="resnet50"
# #ReLU
# SAE="/BS/disentanglement/work/msae/16384_2048_ReLU_003_False_False_0.0_imagenet_resnet50_embeddings.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/fms.py -m $MODEL --sae $SAE  

# SAE="/BS/disentanglement/work/msae/32768_2048_ReLU_003_False_False_0.0_imagenet_resnet50_embeddings.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/fms.py -m $MODEL --sae $SAE  

# SAE="/BS/disentanglement/work/msae/65536_2048_ReLU_003_False_False_0.0_imagenet_resnet50_embeddings.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/fms.py -m $MODEL --sae $SAE  

# #Top256
# SAE="/BS/disentanglement/work/msae/16384_2048_TopKReLU_256_False_False_0.0_imagenet_resnet50_embeddings.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/fms.py -m $MODEL --sae $SAE  

# SAE="/BS/disentanglement/work/msae/32768_2048_TopKReLU_256_False_False_0.0_imagenet_resnet50_embeddings.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/fms.py -m $MODEL --sae $SAE  

# SAE="/BS/disentanglement/work/msae/65536_2048_TopKReLU_256_False_False_0.0_imagenet_resnet50_embeddings.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/fms.py -m $MODEL --sae $SAE  

# #Top64
# SAE="/BS/disentanglement/work/msae/16384_2048_TopKReLU_64_False_False_0.0_imagenet_resnet50_embeddings.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/fms.py -m $MODEL --sae $SAE  

# SAE="/BS/disentanglement/work/msae/32768_2048_TopKReLU_64_False_False_0.0_imagenet_resnet50_embeddings.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/fms.py -m $MODEL --sae $SAE  

# SAE="/BS/disentanglement/work/msae/65536_2048_TopKReLU_64_False_False_0.0_imagenet_resnet50_embeddings.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/fms.py -m $MODEL --sae $SAE  

# #BatchTop256
# SAE="/BS/disentanglement/work/msae/batchtopk/16384_2048_TopKReLU_256_False_False_0.0_imagenet_resnet50_embeddings.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/fms.py -m $MODEL --sae $SAE  

# SAE="/BS/disentanglement/work/msae/batchtopk/32768_2048_TopKReLU_256_False_False_0.0_imagenet_resnet50_embeddings.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/fms.py -m $MODEL --sae $SAE  

# SAE="/BS/disentanglement/work/msae/batchtopk/65536_2048_TopKReLU_256_False_False_0.0_imagenet_resnet50_embeddings.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/fms.py -m $MODEL --sae $SAE  

# #BatchTop64
# SAE="/BS/disentanglement/work/msae/batchtopk/16384_2048_TopKReLU_64_False_False_0.0_imagenet_resnet50_embeddings.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/fms.py -m $MODEL --sae $SAE  

# SAE="/BS/disentanglement/work/msae/batchtopk/32768_2048_TopKReLU_64_False_False_0.0_imagenet_resnet50_embeddings.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/fms.py -m $MODEL --sae $SAE  

# SAE="/BS/disentanglement/work/msae/batchtopk/65536_2048_TopKReLU_64_False_False_0.0_imagenet_resnet50_embeddings.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/fms.py -m $MODEL --sae $SAE  

# #MSAE RW
# SAE="/BS/disentanglement/work/msae/16384_2048_TopKReLU_64_RW_False_False_0.0_imagenet_resnet50_embeddings.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/fms.py -m $MODEL --sae $SAE  

# SAE="/BS/disentanglement/work/msae/32768_2048_TopKReLU_64_RW_False_False_0.0_imagenet_resnet50_embeddings.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/fms.py -m $MODEL --sae $SAE  

# SAE="/BS/disentanglement/work/msae/65536_2048_TopKReLU_64_RW_False_False_0.0_imagenet_resnet50_embeddings.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/fms.py -m $MODEL --sae $SAE  

# #MSAE UW
# SAE="/BS/disentanglement/work/msae/16384_2048_TopKReLU_64_UW_False_False_0.0_imagenet_resnet50_embeddings.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/fms.py -m $MODEL --sae $SAE  

# SAE="/BS/disentanglement/work/msae/32768_2048_TopKReLU_64_UW_False_False_0.0_imagenet_resnet50_embeddings.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/fms.py -m $MODEL --sae $SAE  

# SAE="/BS/disentanglement/work/msae/65536_2048_TopKReLU_64_UW_False_False_0.0_imagenet_resnet50_embeddings.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/fms.py -m $MODEL --sae $SAE  


# echo "ResNet50 SAE evaluated"

# ## DINOv2
# MODEL="dinov2_vitl14"
# # #ReLU
# SAE="/BS/disentanglement/work/msae/8192_1024_ReLU_003_False_False_0.0_imagenet_dinov2_vitl14_embeddings_train.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/fms.py -m $MODEL --sae $SAE  

# SAE="/BS/disentanglement/work/msae/16384_1024_ReLU_003_False_False_0.0_imagenet_dinov2_vitl14_embeddings_train.pth"
                                   
# uv run /BS/disentanglement/work/Disentanglement/MSAE/fms.py -m $MODEL --sae $SAE  

# SAE="/BS/disentanglement/work/msae/32768_1024_ReLU_003_False_False_0.0_imagenet_dinov2_vitl14_embeddings_train.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/fms.py -m $MODEL --sae $SAE  

# #Top256
# SAE="/BS/disentanglement/work/msae/8192_1024_TopKReLU_256_False_False_0.0_imagenet_dinov2_vitl14_embeddings_train.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/fms.py -m $MODEL --sae $SAE  

# SAE="/BS/disentanglement/work/msae/16384_1024_TopKReLU_256_False_False_0.0_imagenet_dinov2_vitl14_embeddings_train.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/fms.py -m $MODEL --sae $SAE  

# SAE="/BS/disentanglement/work/msae/32768_1024_TopKReLU_256_False_False_0.0_imagenet_dinov2_vitl14_embeddings_train.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/fms.py -m $MODEL --sae $SAE  

# #Top64
# SAE="/BS/disentanglement/work/msae/8192_1024_TopKReLU_64_False_False_0.0_imagenet_dinov2_vitl14_embeddings_train.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/fms.py -m $MODEL --sae $SAE  

# SAE="/BS/disentanglement/work/msae/16384_1024_TopKReLU_64_False_False_0.0_imagenet_dinov2_vitl14_embeddings_train.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/fms.py -m $MODEL --sae $SAE  

# SAE="/BS/disentanglement/work/msae/32768_1024_TopKReLU_64_False_False_0.0_imagenet_dinov2_vitl14_embeddings_train.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/fms.py -m $MODEL --sae $SAE  

# #BatchTop256
# SAE="/BS/disentanglement/work/msae/batchtopk/8192_1024_TopKReLU_256_False_False_0.0_imagenet_dinov2_vitl14_embeddings_train.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/fms.py -m $MODEL --sae $SAE  

# SAE="/BS/disentanglement/work/msae/batchtopk/16384_1024_TopKReLU_256_False_False_0.0_imagenet_dinov2_vitl14_embeddings_train.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/fms.py -m $MODEL --sae $SAE  

# SAE="/BS/disentanglement/work/msae/batchtopk/32768_1024_TopKReLU_256_False_False_0.0_imagenet_dinov2_vitl14_embeddings_train.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/fms.py -m $MODEL --sae $SAE  

# #BatchTop64
# SAE="/BS/disentanglement/work/msae/batchtopk/8192_1024_TopKReLU_64_False_False_0.0_imagenet_dinov2_vitl14_embeddings_train.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/fms.py -m $MODEL --sae $SAE  

# SAE="/BS/disentanglement/work/msae/batchtopk/16384_1024_TopKReLU_64_False_False_0.0_imagenet_dinov2_vitl14_embeddings_train.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/fms.py -m $MODEL --sae $SAE  

# SAE="/BS/disentanglement/work/msae/batchtopk/32768_1024_TopKReLU_64_False_False_0.0_imagenet_dinov2_vitl14_embeddings_train.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/fms.py -m $MODEL --sae $SAE  

# #MSAE RW
# SAE="/BS/disentanglement/work/msae/8192_1024_TopKReLU_64_RW_False_False_0.0_imagenet_dinov2_vitl14_embeddings_train.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/fms.py -m $MODEL --sae $SAE  

# SAE="/BS/disentanglement/work/msae/16384_1024_TopKReLU_64_RW_False_False_0.0_imagenet_dinov2_vitl14_embeddings_train.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/fms.py -m $MODEL --sae $SAE  

# SAE="/BS/disentanglement/work/msae/32768_1024_TopKReLU_64_RW_False_False_0.0_imagenet_dinov2_vitl14_embeddings_train.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/fms.py -m $MODEL --sae $SAE  

# #MSAE UW
# SAE="/BS/disentanglement/work/msae/8192_1024_TopKReLU_64_UW_False_False_0.0_imagenet_dinov2_vitl14_embeddings_train.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/fms.py -m $MODEL --sae $SAE  

# SAE="/BS/disentanglement/work/msae/16384_1024_TopKReLU_64_UW_False_False_0.0_imagenet_dinov2_vitl14_embeddings_train.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/fms.py -m $MODEL --sae $SAE  

# SAE="/BS/disentanglement/work/msae/32768_1024_TopKReLU_64_UW_False_False_0.0_imagenet_dinov2_vitl14_embeddings_train.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/fms.py -m $MODEL --sae $SAE  

# echo "DINOv2 evaluated"

# # ##ViTB16
# MODEL="vit_b_16"
# TAU=0.03
# SAE="/BS/disentanglement/work/msae/6144_768_ReLU_003_False_False_0.0_imagenet_vit_b_16_embeddings_train.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/fms.py -m $MODEL --eval-hc True --tau $TAU

# TAU=0.02
# SAE="/BS/disentanglement/work/msae/12288_768_ReLU_003_False_False_0.0_imagenet_vit_b_16_embeddings_train.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/fms.py -m $MODEL --eval-hc True --tau $TAU

# TAU=0.01
# SAE="/BS/disentanglement/work/msae/12288_768_ReLU_003_False_False_0.0_imagenet_vit_b_16_embeddings_train.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/fms.py -m $MODEL --eval-hc True --tau $TAU

# MODEL="resnet50"
# TAU=0.03
# SAE="/BS/disentanglement/work/msae/16384_2048_ReLU_003_False_False_0.0_imagenet_resnet50_embeddings.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/fms.py -m $MODEL --eval-hc True --tau $TAU

# TAU=0.02
# SAE="/BS/disentanglement/work/msae/16384_2048_ReLU_003_False_False_0.0_imagenet_resnet50_embeddings.pth"
# uv run /BS/disentanglement/work/Disentanglement/MSAE/fms.py -m $MODEL --eval-hc True --tau $TAU

#MODEL="resnet50"
#TAU=0.01
#SAE="/BS/disentanglement/work/msae/16384_2048_ReLU_003_False_False_0.0_imagenet_resnet50_embeddings.pth"
#uv run /BS/disentanglement/work/Disentanglement/MSAE/fms.py -m $MODEL --eval-hc True --tau $TAU


MODEL="dinov2_vitl14"
TAU=0.02
uv run /BS/disentanglement/work/Disentanglement/MSAE/fms.py -m $MODEL --eval-hc True --tau $TAU

