# ====================================================
# Train the feature extractor and extract all features
# ====================================================
CUDA_VISIBLE_DEVICES=? python3 /home/cclin/few_shot_learning/sem_hal/train_extractor.py \
    --data_path /data/put_data/cclin/datasets/awa/Animals_with_Attributes2-split-cv-b15n10 \
    --result_path /home/cclin/few_shot_learning/sem_hal/awa/results_cv \
    --extractor_name VGG_EXT_b64_lr5e6_ep500_fc256_l2reg1e3_p20 \
    --vgg16_npy_path /data/put_data/cclin/ntu/dlcv2018/hw3/vgg16.npy \
    --n_base_classes 15 \
    --n_top 5 \
    --bsize 64 \
    --learning_rate 5e-6 \
    --l2scale 1e-3 \
    --num_epoch 500 \
    --img_size_h 64 \
    --img_size_w 64 \
    --fc_dim 256 \
    --patience 20 \
    --debug

CUDA_VISIBLE_DEVICES=? python3 /home/cclin/few_shot_learning/sem_hal/train_extractor.py \
    --data_path /data/put_data/cclin/datasets/awa/Animals_with_Attributes2-split-final-b15n10 \
    --result_path /home/cclin/few_shot_learning/sem_hal/awa/results_final \
    --extractor_name VGG_EXT_b64_lr5e6_ep500_fc256_l2reg1e3_p20 \
    --vgg16_npy_path /data/put_data/cclin/ntu/dlcv2018/hw3/vgg16.npy \
    --n_base_classes 15 \
    --n_top 5 \
    --bsize 64 \
    --learning_rate 5e-6 \
    --l2scale 1e-3 \
    --num_epoch 500 \
    --img_size_h 64 \
    --img_size_w 64 \
    --fc_dim 256 \
    --patience 20 \
    --debug

# ===================================
# Train hallucinator by meta-learning
# ===================================
# All the following scripts take 3 arguments:
# (1) 'cv' of 'final' to specify the result folder;
# (2) m_support: number of class in the support set;
# (3) n_support: number of shot per class in the support set.
# ----------------------------
# Train GAN-based hallucinator
# ---------------------------- 
CUDA_VISIBLE_DEVICES=? sh script_hal_PN_GAN.sh cv 15 5
CUDA_VISIBLE_DEVICES=? sh script_hal_PN_GAN.sh cv 15 10
CUDA_VISIBLE_DEVICES=? sh script_hal_PN_GAN.sh final 15 5
CUDA_VISIBLE_DEVICES=? sh script_hal_PN_GAN.sh final 15 10

# -------------------------------------------------------------------------------------
# Train GAN2-based hallucinator (adds one more layer to the hallucinator of HAL_PN_GAN)
# -------------------------------------------------------------------------------------
CUDA_VISIBLE_DEVICES=? sh script_hal_PN_GAN2.sh cv 15 5
CUDA_VISIBLE_DEVICES=? sh script_hal_PN_GAN2.sh cv 15 10
CUDA_VISIBLE_DEVICES=? sh script_hal_PN_GAN2.sh final 15 5
CUDA_VISIBLE_DEVICES=? sh script_hal_PN_GAN2.sh final 15 10

# ---------------------------------------------
# Train semantics-guided GAN-based hallucinator
# ---------------------------------------------
CUDA_VISIBLE_DEVICES=? sh script_hal_PN_VAEGAN2.sh cv 15 5
CUDA_VISIBLE_DEVICES=? sh script_hal_PN_VAEGAN2.sh cv 15 10
CUDA_VISIBLE_DEVICES=? sh script_hal_PN_VAEGAN2.sh final 15 5
CUDA_VISIBLE_DEVICES=? sh script_hal_PN_VAEGAN2.sh final 15 10

# =======================
# Few-shot learning on cv
# =======================
# All the following scripts take 1 argument: number of shot
# All scripts run 10 iterations since accuracies are too small and variances are too large
# ---------------------------
# Baseline (no hallucination)
# ---------------------------
CUDA_VISIBLE_DEVICES=? sh script_cv_nohal.sh 01 > ./awa/results_cv/results_summary_shot01_nohal
CUDA_VISIBLE_DEVICES=? sh script_cv_nohal.sh 02 > ./awa/results_cv/results_summary_shot02_nohal
CUDA_VISIBLE_DEVICES=? sh script_cv_nohal.sh 05 > ./awa/results_cv/results_summary_shot05_nohal
# Parse results (saved as csv files)
egrep 'WARNING: the output path|top-5 test accuracy' ./awa/results_cv/results_summary_shot01_nohal > ./awa/results_cv/results_summary_shot01_nohal_acc
python3 acc_parser_from0.py ./awa/results_cv/results_summary_shot01_nohal_acc > ./awa/results_cv/results_summary_shot01_nohal_acc.csv
egrep 'WARNING: the output path|top-5 test accuracy' ./awa/results_cv/results_summary_shot02_nohal > ./awa/results_cv/results_summary_shot02_nohal_acc
python3 acc_parser_from0.py ./awa/results_cv/results_summary_shot02_nohal_acc > ./awa/results_cv/results_summary_shot02_nohal_acc.csv
egrep 'WARNING: the output path|top-5 test accuracy' ./awa/results_cv/results_summary_shot05_nohal > ./awa/results_cv/results_summary_shot05_nohal_acc
python3 acc_parser_from0.py ./awa/results_cv/results_summary_shot05_nohal_acc > ./awa/results_cv/results_summary_shot05_nohal_acc.csv

# -----------------------
# GAN-based hallucination
# -----------------------
CUDA_VISIBLE_DEVICES=? sh script_cv_PN_GAN.sh 01 > ./awa/results_cv/results_summary_shot01_GAN
CUDA_VISIBLE_DEVICES=? sh script_cv_PN_GAN.sh 02 > ./awa/results_cv/results_summary_shot02_GAN
CUDA_VISIBLE_DEVICES=? sh script_cv_PN_GAN.sh 05 > ./awa/results_cv/results_summary_shot05_GAN
# Parse results (saved as csv files)
egrep 'WARNING: the output path|top-5 test accuracy' ./awa/results_cv/results_summary_shot01_GAN > ./awa/results_cv/results_summary_shot01_GAN_acc
python3 acc_parser_from0.py ./awa/results_cv/results_summary_shot01_GAN_acc > ./awa/results_cv/results_summary_shot01_GAN_acc.csv
egrep 'WARNING: the output path|top-5 test accuracy' ./awa/results_cv/results_summary_shot02_GAN > ./awa/results_cv/results_summary_shot02_GAN_acc
python3 acc_parser_from0.py ./awa/results_cv/results_summary_shot02_GAN_acc > ./awa/results_cv/results_summary_shot02_GAN_acc.csv
egrep 'WARNING: the output path|top-5 test accuracy' ./awa/results_cv/results_summary_shot05_GAN > ./awa/results_cv/results_summary_shot05_GAN_acc
python3 acc_parser_from0.py ./awa/results_cv/results_summary_shot05_GAN_acc > ./awa/results_cv/results_summary_shot05_GAN_acc.csv

# --------------------------------------------------------------------------------
# GAN2-based hallucination (adds one more layer to the hallucinator of HAL_PN_GAN)
# --------------------------------------------------------------------------------
CUDA_VISIBLE_DEVICES=? sh script_cv_PN_GAN2.sh 01 > ./awa/results_cv/results_summary_shot01_GAN2
CUDA_VISIBLE_DEVICES=? sh script_cv_PN_GAN2.sh 02 > ./awa/results_cv/results_summary_shot02_GAN2
CUDA_VISIBLE_DEVICES=? sh script_cv_PN_GAN2.sh 05 > ./awa/results_cv/results_summary_shot05_GAN2
# Parse results (saved as csv files)
egrep 'WARNING: the output path|top-5 test accuracy' ./awa/results_cv/results_summary_shot01_GAN2 > ./awa/results_cv/results_summary_shot01_GAN2_acc
python3 acc_parser_from0.py ./awa/results_cv/results_summary_shot01_GAN2_acc > ./awa/results_cv/results_summary_shot01_GAN2_acc.csv
egrep 'WARNING: the output path|top-5 test accuracy' ./awa/results_cv/results_summary_shot02_GAN2 > ./awa/results_cv/results_summary_shot02_GAN2_acc
python3 acc_parser_from0.py ./awa/results_cv/results_summary_shot02_GAN2_acc > ./awa/results_cv/results_summary_shot02_GAN2_acc.csv
egrep 'WARNING: the output path|top-5 test accuracy' ./awa/results_cv/results_summary_shot05_GAN2 > ./awa/results_cv/results_summary_shot05_GAN2_acc
python3 acc_parser_from0.py ./awa/results_cv/results_summary_shot05_GAN2_acc > ./awa/results_cv/results_summary_shot05_GAN2_acc.csv

# -------------------------------------------------------------------------------------
# Semantics-guided GAN-based hallucination
# (implementation of our idea (C.-C. Lin, ICIP 2019) with a simpler version of encoder)
# -------------------------------------------------------------------------------------
CUDA_VISIBLE_DEVICES=? sh script_cv_PN_VAEGAN2.sh 01 > ./awa/results_cv/results_summary_shot01_VAEGAN2
CUDA_VISIBLE_DEVICES=? sh script_cv_PN_VAEGAN2.sh 02 > ./awa/results_cv/results_summary_shot02_VAEGAN2
CUDA_VISIBLE_DEVICES=? sh script_cv_PN_VAEGAN2.sh 05 > ./awa/results_cv/results_summary_shot05_VAEGAN2
# Parse results (saved as csv files)
egrep 'WARNING: the output path|top-5 test accuracy' ./awa/results_cv/results_summary_shot01_VAEGAN2 > ./awa/results_cv/results_summary_shot01_VAEGAN2_acc
python3 acc_parser_from0.py ./awa/results_cv/results_summary_shot01_VAEGAN2_acc > ./awa/results_cv/results_summary_shot01_VAEGAN2_acc.csv
egrep 'WARNING: the output path|top-5 test accuracy' ./awa/results_cv/results_summary_shot02_VAEGAN2 > ./awa/results_cv/results_summary_shot02_VAEGAN2_acc
python3 acc_parser_from0.py ./awa/results_cv/results_summary_shot02_VAEGAN2_acc > ./awa/results_cv/results_summary_shot02_VAEGAN2_acc.csv
egrep 'WARNING: the output path|top-5 test accuracy' ./awa/results_cv/results_summary_shot05_VAEGAN2 > ./awa/results_cv/results_summary_shot05_VAEGAN2_acc
python3 acc_parser_from0.py ./awa/results_cv/results_summary_shot05_VAEGAN2_acc > ./awa/results_cv/results_summary_shot05_VAEGAN2_acc.csv

# ==========================
# Few-shot learning on final
# ==========================
# Extract best hyper-parameters according on the results on cv
# ---------------------------
# Baseline (no hallucination)
# ---------------------------
# make training scripts by extracting the best hyper-parameters
python3 extract_hyper.py ./awa/results_cv/results_summary_shot01_nohal_acc.csv nohal > script_final_shot01_nohal.sh
python3 extract_hyper.py ./awa/results_cv/results_summary_shot02_nohal_acc.csv nohal > script_final_shot02_nohal.sh
python3 extract_hyper.py ./awa/results_cv/results_summary_shot05_nohal_acc.csv nohal > script_final_shot05_nohal.sh
# run 10 iterations of FSL
CUDA_VISIBLE_DEVICES=? sh script_final_shot01_nohal.sh > ./awa/results_final/results_summary_shot01_nohal
CUDA_VISIBLE_DEVICES=? sh script_final_shot02_nohal.sh > ./awa/results_final/results_summary_shot02_nohal
CUDA_VISIBLE_DEVICES=? sh script_final_shot05_nohal.sh > ./awa/results_final/results_summary_shot05_nohal
# Parse results
egrep 'WARNING: the output path|top-5 test accuracy' ./awa/results_final/results_summary_shot01_nohal > ./awa/results_final/results_summary_shot01_nohal_acc
python3 acc_parser_from0.py ./awa/results_final/results_summary_shot01_nohal_acc
egrep 'WARNING: the output path|top-5 test accuracy' ./awa/results_final/results_summary_shot02_nohal > ./awa/results_final/results_summary_shot02_nohal_acc
python3 acc_parser_from0.py ./awa/results_final/results_summary_shot02_nohal_acc
egrep 'WARNING: the output path|top-5 test accuracy' ./awa/results_final/results_summary_shot05_nohal > ./awa/results_final/results_summary_shot05_nohal_acc
python3 acc_parser_from0.py ./awa/results_final/results_summary_shot05_nohal_acc

# -----------------------
# GAN-based hallucination
# -----------------------
python3 extract_hyper.py ./awa/results_cv/results_summary_shot01_GAN_acc.csv PN_GAN > script_final_shot01_GAN.sh
python3 extract_hyper.py ./awa/results_cv/results_summary_shot02_GAN_acc.csv PN_GAN > script_final_shot02_GAN.sh
python3 extract_hyper.py ./awa/results_cv/results_summary_shot05_GAN_acc.csv PN_GAN > script_final_shot05_GAN.sh
# run 10 iterations of FSL
CUDA_VISIBLE_DEVICES=? sh script_final_shot01_GAN.sh > ./awa/results_final/results_summary_shot01_GAN
CUDA_VISIBLE_DEVICES=? sh script_final_shot02_GAN.sh > ./awa/results_final/results_summary_shot02_GAN
CUDA_VISIBLE_DEVICES=? sh script_final_shot05_GAN.sh > ./awa/results_final/results_summary_shot05_GAN
# Parse results
egrep 'WARNING: the output path|top-5 test accuracy' ./awa/results_final/results_summary_shot01_GAN > ./awa/results_final/results_summary_shot01_GAN_acc
python3 acc_parser_from0.py ./awa/results_final/results_summary_shot01_GAN_acc
egrep 'WARNING: the output path|top-5 test accuracy' ./awa/results_final/results_summary_shot02_GAN > ./awa/results_final/results_summary_shot02_GAN_acc
python3 acc_parser_from0.py ./awa/results_final/results_summary_shot02_GAN_acc
egrep 'WARNING: the output path|top-5 test accuracy' ./awa/results_final/results_summary_shot05_GAN > ./awa/results_final/results_summary_shot05_GAN_acc
python3 acc_parser_from0.py ./awa/results_final/results_summary_shot05_GAN_acc

# --------------------------------------------------------------------------------
# GAN2-based hallucination (adds one more layer to the hallucinator of HAL_PN_GAN)
# --------------------------------------------------------------------------------
python3 extract_hyper.py ./awa/results_cv/results_summary_shot01_GAN2_acc.csv PN_GAN2 > script_final_shot01_GAN2.sh
python3 extract_hyper.py ./awa/results_cv/results_summary_shot02_GAN2_acc.csv PN_GAN2 > script_final_shot02_GAN2.sh
python3 extract_hyper.py ./awa/results_cv/results_summary_shot05_GAN2_acc.csv PN_GAN2 > script_final_shot05_GAN2.sh
# run 10 iterations of FSL
CUDA_VISIBLE_DEVICES=? sh script_final_shot01_GAN2.sh > ./awa/results_final/results_summary_shot01_GAN2
CUDA_VISIBLE_DEVICES=? sh script_final_shot02_GAN2.sh > ./awa/results_final/results_summary_shot02_GAN2
CUDA_VISIBLE_DEVICES=? sh script_final_shot05_GAN2.sh > ./awa/results_final/results_summary_shot05_GAN2
# Parse results
egrep 'WARNING: the output path|top-5 test accuracy' ./awa/results_final/results_summary_shot01_GAN2 > ./awa/results_final/results_summary_shot01_GAN2_acc
python3 acc_parser_from0.py ./awa/results_final/results_summary_shot01_GAN2_acc
egrep 'WARNING: the output path|top-5 test accuracy' ./awa/results_final/results_summary_shot02_GAN2 > ./awa/results_final/results_summary_shot02_GAN2_acc
python3 acc_parser_from0.py ./awa/results_final/results_summary_shot02_GAN2_acc
egrep 'WARNING: the output path|top-5 test accuracy' ./awa/results_final/results_summary_shot05_GAN2 > ./awa/results_final/results_summary_shot05_GAN2_acc
python3 acc_parser_from0.py ./awa/results_final/results_summary_shot05_GAN2_acc

# -------------------------------------------------------------------------------------
# Semantics-guided GAN-based hallucination
# (implementation of our idea (C.-C. Lin, ICIP 2019) with a simpler version of encoder)
# -------------------------------------------------------------------------------------
python3 extract_hyper.py ./awa/results_cv/results_summary_shot01_VAEGAN2_acc.csv PN_VAEGAN2 > script_final_shot01_VAEGAN2.sh
python3 extract_hyper.py ./awa/results_cv/results_summary_shot02_VAEGAN2_acc.csv PN_VAEGAN2 > script_final_shot02_VAEGAN2.sh
python3 extract_hyper.py ./awa/results_cv/results_summary_shot05_VAEGAN2_acc.csv PN_VAEGAN2 > script_final_shot05_VAEGAN2.sh
# run 10 iterations of FSL
CUDA_VISIBLE_DEVICES=? sh script_final_shot01_VAEGAN2.sh > ./awa/results_final/results_summary_shot01_VAEGAN2
CUDA_VISIBLE_DEVICES=? sh script_final_shot02_VAEGAN2.sh > ./awa/results_final/results_summary_shot02_VAEGAN2
CUDA_VISIBLE_DEVICES=? sh script_final_shot05_VAEGAN2.sh > ./awa/results_final/results_summary_shot05_VAEGAN2
# Parse results
egrep 'WARNING: the output path|top-5 test accuracy' ./awa/results_final/results_summary_shot01_VAEGAN2 > ./awa/results_final/results_summary_shot01_VAEGAN2_acc
python3 acc_parser_from0.py ./awa/results_final/results_summary_shot01_VAEGAN2_acc
egrep 'WARNING: the output path|top-5 test accuracy' ./awa/results_final/results_summary_shot02_VAEGAN2 > ./awa/results_final/results_summary_shot02_VAEGAN2_acc
python3 acc_parser_from0.py ./awa/results_final/results_summary_shot02_VAEGAN2_acc
egrep 'WARNING: the output path|top-5 test accuracy' ./awa/results_final/results_summary_shot05_VAEGAN2 > ./awa/results_final/results_summary_shot05_VAEGAN2_acc
python3 acc_parser_from0.py ./awa/results_final/results_summary_shot05_VAEGAN2_acc