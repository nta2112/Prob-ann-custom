#!/usr/bin/env bash

# Vô hiệu hóa wandb để tránh việc yêu cầu đăng nhập tương tác gây kẹt tiến trình
export WANDB_MODE=disabled

# Kịch bản chạy huấn luyện 1 Epoch rồi chạy đánh giá (test) ngay lập tức cho từng Task (Task 1 -> Task 4).
# Sử dụng: bash tools/run_train_eval_kaggle.sh <đường_dẫn_checkpoint_ban_đầu> <batch_size_train> <batch_size_eval>
# Mặc định:
#   Checkpoint ban đầu: /kaggle/input/prob-checkpoints
#   Batch size train: 4 (mỗi GPU)
#   Batch size eval: 16 (mỗi GPU)

CKPT_DIR=${1:-"/kaggle/input/prob-checkpoints"}
TRAIN_BATCH_SIZE=${2:-4}
EVAL_BATCH_SIZE=${3:-16}
DATA_ROOT="/kaggle/working/datasets/IP102"
EXP_DIR="/kaggle/working/exps/IP102"
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

echo "=========================================================================="
echo "BẮT ĐẦU QUY TRÌNH HUẤN LUYỆN 1 EPOCH & ĐÁNH GIÁ TỪNG TASK (TASK 1 -> TASK 4)"
echo "Checkpoints Directory: ${CKPT_DIR}"
echo "Train Batch Size per GPU: ${TRAIN_BATCH_SIZE}"
echo "Eval Batch Size per GPU: ${EVAL_BATCH_SIZE}"
echo "Data Root: ${DATA_ROOT}"
echo "W&B Mode: ${WANDB_MODE}"
echo "=========================================================================="

# ======================= TASK 1 =======================
echo "----------------------------------------------------"
echo ">>> [TASK 1] Huấn luyện 1 Epoch (Epoch 0)"
python -m torch.distributed.run --nproc_per_node=2 --master_port=29501 main_open_world.py \
    --output_dir "${EXP_DIR}/t1" --dataset IP102 --num_classes 103 --PREV_INTRODUCED_CLS 0 --CUR_INTRODUCED_CLS 27 \
    --train_set "owod_t1_train" --test_set "owod_all_task_test" --epochs 1 \
    --model_type "prob" --obj_loss_coef 8e-4 --obj_temp 1.3 \
    --batch_size ${TRAIN_BATCH_SIZE} --num_workers ${NUM_WORKERS} \
    --pretrain "${CKPT_DIR}/t1.pth" --data_root ${DATA_ROOT} \
    --exemplar_replay_selection --exemplar_replay_max_length 850 \
    --exemplar_replay_dir "IP102_ft" --exemplar_replay_cur_file "learned_owod_t1_ft.txt" \
    --wandb_project ""

echo ">>> [TASK 1] Đánh giá kiểm thử (Test) trên Lớp 1 - 27"
python -m torch.distributed.run --nproc_per_node=2 --master_port=29502 main_open_world.py \
    --output_dir "${EXP_DIR}/t1/eval" --dataset IP102 --num_classes 103 --PREV_INTRODUCED_CLS 0 --CUR_INTRODUCED_CLS 27 \
    --train_set "owod_t1_train" --test_set "owod_all_task_test" --epochs 1 \
    --model_type "prob" --obj_loss_coef 8e-4 --obj_temp 1.3 \
    --batch_size ${EVAL_BATCH_SIZE} --num_workers ${NUM_WORKERS} \
    --pretrain "${EXP_DIR}/t1/checkpoint0000.pth" --data_root ${DATA_ROOT} \
    --eval --wandb_project ""


# ======================= TASK 2 =======================
echo "----------------------------------------------------"
echo ">>> [TASK 2] Huấn luyện 1 Epoch (Epoch 1)"
python -m torch.distributed.run --nproc_per_node=2 --master_port=29503 main_open_world.py \
    --output_dir "${EXP_DIR}/t2" --dataset IP102 --num_classes 103 --PREV_INTRODUCED_CLS 27 --CUR_INTRODUCED_CLS 25 \
    --train_set "owod_t2_train" --test_set "owod_all_task_test" --epochs 2 \
    --model_type "prob" --obj_loss_coef 8e-4 --obj_temp 1.3 --freeze_prob_model \
    --batch_size ${TRAIN_BATCH_SIZE} --num_workers ${NUM_WORKERS} \
    --pretrain "${EXP_DIR}/t1/checkpoint0000.pth" --data_root ${DATA_ROOT} --lr 2e-5 \
    --exemplar_replay_selection --exemplar_replay_max_length 1743 --exemplar_replay_dir "IP102_ft" \
    --exemplar_replay_prev_file "learned_owod_t1_ft.txt" --exemplar_replay_cur_file "learned_owod_t2_ft.txt" \
    --wandb_project ""

echo ">>> [TASK 2] Tinh chỉnh mẫu (Finetune Replay) - 1 Epoch (Epoch 2)"
python -m torch.distributed.run --nproc_per_node=2 --master_port=29504 main_open_world.py \
    --output_dir "${EXP_DIR}/t2_ft" --dataset IP102 --num_classes 103 --PREV_INTRODUCED_CLS 27 --CUR_INTRODUCED_CLS 25 \
    --train_set "IP102_ft/learned_owod_t2_ft" --test_set "owod_all_task_test" --epochs 3 \
    --model_type "prob" --obj_loss_coef 8e-4 --obj_temp 1.3 \
    --batch_size ${TRAIN_BATCH_SIZE} --num_workers ${NUM_WORKERS} \
    --pretrain "${EXP_DIR}/t2/checkpoint0001.pth" --data_root ${DATA_ROOT} \
    --wandb_project ""

echo ">>> [TASK 2] Đánh giá kiểm thử (Test) trên Lớp 1 - 52"
python -m torch.distributed.run --nproc_per_node=2 --master_port=29505 main_open_world.py \
    --output_dir "${EXP_DIR}/t2_ft/eval" --dataset IP102 --num_classes 103 --PREV_INTRODUCED_CLS 27 --CUR_INTRODUCED_CLS 25 \
    --train_set "owod_t2_train" --test_set "owod_all_task_test" --epochs 3 \
    --model_type "prob" --obj_loss_coef 8e-4 --obj_temp 1.3 \
    --batch_size ${EVAL_BATCH_SIZE} --num_workers ${NUM_WORKERS} \
    --pretrain "${EXP_DIR}/t2_ft/checkpoint0002.pth" --data_root ${DATA_ROOT} \
    --eval --wandb_project ""


# ======================= TASK 3 =======================
echo "----------------------------------------------------"
echo ">>> [TASK 3] Huấn luyện 1 Epoch (Epoch 3)"
python -m torch.distributed.run --nproc_per_node=2 --master_port=29506 main_open_world.py \
    --output_dir "${EXP_DIR}/t3" --dataset IP102 --num_classes 103 --PREV_INTRODUCED_CLS 52 --CUR_INTRODUCED_CLS 25 \
    --train_set "owod_t3_train" --test_set "owod_all_task_test" --epochs 4 \
    --model_type "prob" --obj_loss_coef 8e-4 --obj_temp 1.3 --freeze_prob_model \
    --batch_size ${TRAIN_BATCH_SIZE} --num_workers ${NUM_WORKERS} \
    --pretrain "${EXP_DIR}/t2_ft/checkpoint0002.pth" --data_root ${DATA_ROOT} --lr 2e-5 \
    --exemplar_replay_selection --exemplar_replay_max_length 2361 --exemplar_replay_dir "IP102_ft" \
    --exemplar_replay_prev_file "learned_owod_t2_ft.txt" --exemplar_replay_cur_file "learned_owod_t3_ft.txt" \
    --wandb_project ""

echo ">>> [TASK 3] Tinh chỉnh mẫu (Finetune Replay) - 1 Epoch (Epoch 4)"
python -m torch.distributed.run --nproc_per_node=2 --master_port=29507 main_open_world.py \
    --output_dir "${EXP_DIR}/t3_ft" --dataset IP102 --num_classes 103 --PREV_INTRODUCED_CLS 52 --CUR_INTRODUCED_CLS 25 \
    --train_set "IP102_ft/learned_owod_t3_ft" --test_set "owod_all_task_test" --epochs 5 \
    --model_type "prob" --obj_loss_coef 8e-4 --obj_temp 1.3 \
    --batch_size ${TRAIN_BATCH_SIZE} --num_workers ${NUM_WORKERS} \
    --pretrain "${EXP_DIR}/t3/checkpoint0003.pth" --data_root ${DATA_ROOT} \
    --wandb_project ""

echo ">>> [TASK 3] Đánh giá kiểm thử (Test) trên Lớp 1 - 77"
python -m torch.distributed.run --nproc_per_node=2 --master_port=29508 main_open_world.py \
    --output_dir "${EXP_DIR}/t3_ft/eval" --dataset IP102 --num_classes 103 --PREV_INTRODUCED_CLS 52 --CUR_INTRODUCED_CLS 25 \
    --train_set "owod_t3_train" --test_set "owod_all_task_test" --epochs 5 \
    --model_type "prob" --obj_loss_coef 8e-4 --obj_temp 1.3 \
    --batch_size ${EVAL_BATCH_SIZE} --num_workers ${NUM_WORKERS} \
    --pretrain "${EXP_DIR}/t3_ft/checkpoint0004.pth" --data_root ${DATA_ROOT} \
    --eval --wandb_project ""


# ======================= TASK 4 =======================
echo "----------------------------------------------------"
echo ">>> [TASK 4] Huấn luyện 1 Epoch (Epoch 5)"
python -m torch.distributed.run --nproc_per_node=2 --master_port=29509 main_open_world.py \
    --output_dir "${EXP_DIR}/t4" --dataset IP102 --num_classes 103 --PREV_INTRODUCED_CLS 77 --CUR_INTRODUCED_CLS 25 \
    --train_set "owod_t4_train" --test_set "owod_all_task_test" --epochs 6 \
    --model_type "prob" --obj_loss_coef 8e-4 --obj_temp 1.3 --freeze_prob_model \
    --batch_size ${TRAIN_BATCH_SIZE} --num_workers ${NUM_WORKERS} \
    --pretrain "${EXP_DIR}/t3_ft/checkpoint0004.pth" --data_root ${DATA_ROOT} --lr 2e-5 \
    --exemplar_replay_selection --exemplar_replay_max_length 2749 --exemplar_replay_dir "IP102_ft" \
    --exemplar_replay_prev_file "learned_owod_t3_ft.txt" --exemplar_replay_cur_file "learned_owod_t4_ft.txt" \
    --num_inst_per_class 40 --wandb_project ""

echo ">>> [TASK 4] Tinh chỉnh mẫu (Finetune Replay) - 1 Epoch (Epoch 6)"
python -m torch.distributed.run --nproc_per_node=2 --master_port=29510 main_open_world.py \
    --output_dir "${EXP_DIR}/t4_ft" --dataset IP102 --num_classes 103 --PREV_INTRODUCED_CLS 77 --CUR_INTRODUCED_CLS 25 \
    --train_set "IP102_ft/learned_owod_t4_ft" --test_set "owod_all_task_test" --epochs 7 \
    --model_type "prob" --obj_loss_coef 8e-4 --obj_temp 1.3 \
    --batch_size ${TRAIN_BATCH_SIZE} --num_workers ${NUM_WORKERS} \
    --pretrain "${EXP_DIR}/t4/checkpoint0005.pth" --data_root ${DATA_ROOT} \
    --wandb_project ""

echo ">>> [TASK 4] Đánh giá kiểm thử (Test) trên Lớp 1 - 102"
python -m torch.distributed.run --nproc_per_node=2 --master_port=29511 main_open_world.py \
    --output_dir "${EXP_DIR}/t4_ft/eval" --dataset IP102 --num_classes 103 --PREV_INTRODUCED_CLS 77 --CUR_INTRODUCED_CLS 25 \
    --train_set "owod_t4_train" --test_set "owod_all_task_test" --epochs 7 \
    --model_type "prob" --obj_loss_coef 8e-4 --obj_temp 1.3 \
    --batch_size ${EVAL_BATCH_SIZE} --num_workers ${NUM_WORKERS} \
    --pretrain "${EXP_DIR}/t4_ft/checkpoint0006.pth" --data_root ${DATA_ROOT} \
    --eval --wandb_project ""

echo "=========================================================================="
echo "QUÁ TRÌNH HUẤN LUYỆN 1 EPOCH & ĐÁNH GIÁ HOÀN THÀNH!"
echo "Các checkpoint lưu tại /kaggle/working/exps/IP102"
echo "Các kết quả đánh giá lưu trong thư mục eval của từng task"
echo "=========================================================================="
