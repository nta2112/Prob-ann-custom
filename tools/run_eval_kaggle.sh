#!/usr/bin/env bash

# Vô hiệu hóa wandb để tránh việc yêu cầu đăng nhập tương tác gây kẹt tiến trình
export WANDB_MODE=disabled

# Kịch bản chạy đánh giá mô hình PROB với tập dữ liệu IP102 trên Kaggle dùng 2 GPU.
# Sử dụng: bash tools/run_eval_kaggle.sh <đường_dẫn_thư_mục_chứa_checkpoints> <batch_size_mỗi_gpu>
# Mặc định:
#   Checkpoints: /kaggle/input/prob-checkpoints (hoặc đường dẫn bất kỳ do người dùng cấu hình)
#   Batch size: 8 (mỗi GPU)

CKPT_DIR=${1:-"/kaggle/input/prob-checkpoints"}
BATCH_SIZE=${2:-8}
DATA_ROOT="/kaggle/working/datasets/IP102"
NUM_WORKERS=2

# Tự động tìm thư mục chứa checkpoint nếu không tìm thấy t1.pth ở đường dẫn mặc định
if [ ! -d "${CKPT_DIR}" ] || [ ! -f "${CKPT_DIR}/t1.pth" ]; then
    echo "Warning: Checkpoint directory '${CKPT_DIR}' or '${CKPT_DIR}/t1.pth' not found."
    echo "Searching for t1.pth in /kaggle/input..."
    FOUND_PATH=$(find /kaggle/input -name "t1.pth" -print -quit 2>/dev/null)
    if [ -n "${FOUND_PATH}" ]; then
        CKPT_DIR=$(dirname "${FOUND_PATH}")
        echo "Found checkpoint directory: ${CKPT_DIR}"
    else
        echo "Could not find t1.pth in /kaggle/input. Proceeding with default."
    fi
fi

echo "============================================="
echo "BẮT ĐẦU ĐÁNH GIÁ MÔ HÌNH PROB TRÊN KAGGLE (2 GPU)"
echo "Checkpoints Directory: ${CKPT_DIR}"
echo "Batch Size per GPU: ${BATCH_SIZE}"
echo "Data Root: ${DATA_ROOT}"
echo "W&B Mode: ${WANDB_MODE}"
echo "============================================="

# ----------------- TASK 1 -----------------
echo ">>> [TASK 1] Đánh giá trên Lớp 1 - 27 (27 Classes)"
python -m torch.distributed.run --nproc_per_node=2 --master_port=29505 main_open_world.py \
    --output_dir "/kaggle/working/results_ip102/t1" \
    --dataset IP102 \
    --num_classes 103 \
    --PREV_INTRODUCED_CLS 0 \
    --CUR_INTRODUCED_CLS 27 \
    --train_set "owod_t1_train" \
    --test_set "owod_all_task_test" \
    --model_type "prob" \
    --obj_loss_coef 8e-4 \
    --obj_temp 1.3 \
    --batch_size ${BATCH_SIZE} \
    --num_workers ${NUM_WORKERS} \
    --pretrain "${CKPT_DIR}/t1.pth" \
    --data_root ${DATA_ROOT} \
    --eval --wandb_project ""

# ----------------- TASK 2 -----------------
echo ">>> [TASK 2] Đánh giá trên Lớp 1 - 52 (Thêm 25 Classes)"
python -m torch.distributed.run --nproc_per_node=2 --master_port=29506 main_open_world.py \
    --output_dir "/kaggle/working/results_ip102/t2" \
    --dataset IP102 \
    --num_classes 103 \
    --PREV_INTRODUCED_CLS 27 \
    --CUR_INTRODUCED_CLS 25 \
    --train_set "owod_t2_train" \
    --test_set "owod_all_task_test" \
    --model_type "prob" \
    --obj_loss_coef 8e-4 \
    --obj_temp 1.3 \
    --batch_size ${BATCH_SIZE} \
    --num_workers ${NUM_WORKERS} \
    --pretrain "${CKPT_DIR}/t2.pth" \
    --data_root ${DATA_ROOT} \
    --eval --wandb_project ""

# ----------------- TASK 3 -----------------
echo ">>> [TASK 3] Đánh giá trên Lớp 1 - 77 (Thêm 25 Classes)"
python -m torch.distributed.run --nproc_per_node=2 --master_port=29507 main_open_world.py \
    --output_dir "/kaggle/working/results_ip102/t3" \
    --dataset IP102 \
    --num_classes 103 \
    --PREV_INTRODUCED_CLS 52 \
    --CUR_INTRODUCED_CLS 25 \
    --train_set "owod_t3_train" \
    --test_set "owod_all_task_test" \
    --model_type "prob" \
    --obj_loss_coef 8e-4 \
    --obj_temp 1.3 \
    --batch_size ${BATCH_SIZE} \
    --num_workers ${NUM_WORKERS} \
    --pretrain "${CKPT_DIR}/t3.pth" \
    --data_root ${DATA_ROOT} \
    --eval --wandb_project ""

# ----------------- TASK 4 -----------------
echo ">>> [TASK 4] Đánh giá trên Lớp 1 - 102 (Thêm 25 Classes)"
python -m torch.distributed.run --nproc_per_node=2 --master_port=29508 main_open_world.py \
    --output_dir "/kaggle/working/results_ip102/t4" \
    --dataset IP102 \
    --num_classes 103 \
    --PREV_INTRODUCED_CLS 77 \
    --CUR_INTRODUCED_CLS 25 \
    --train_set "owod_t4_train" \
    --test_set "owod_all_task_test" \
    --model_type "prob" \
    --obj_loss_coef 8e-4 \
    --obj_temp 1.3 \
    --batch_size ${BATCH_SIZE} \
    --num_workers ${NUM_WORKERS} \
    --pretrain "${CKPT_DIR}/t4.pth" \
    --data_root ${DATA_ROOT} \
    --eval --wandb_project ""

echo "============================================="
echo "QUÁ TRÌNH ĐÁNH GIÁ ĐÃ HOÀN THÀNH!"
echo "Các file kết quả được lưu tại /kaggle/working/results_ip102/"
echo "============================================="
