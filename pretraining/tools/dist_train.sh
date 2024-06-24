# !/usr/bin/env bash

CONFIG=$1
GPUS=$2
# CONFIG2=$2
# GPUS=$3
# CONFIG1=$2
# CONFIG2=$3
# CONFIG3=$4
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/train.py \
    $CONFIG \
    --seed 0 \
    --launcher pytorch

# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# python -m torch.distributed.launch \
#     --nnodes=$NNODES \
#     --node_rank=$NODE_RANK \
#     --master_addr=$MASTER_ADDR \
#     --nproc_per_node=$GPUS \
#     --master_port=$PORT \
#     $(dirname "$0")/train.py \
#     $CONFIG2 \
#     --seed 0 \
#     --launcher pytorch

# # PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# python -m torch.distributed.launch \
#     --nnodes=$NNODES \
#     --node_rank=$NODE_RANK \
#     --master_addr=$MASTER_ADDR \
#     --nproc_per_node=$GPUS \
#     --master_port=$PORT \
#     $(dirname "$0")/train.py \
#     $CONFIG3 \
#     --seed 0 \
#     --launcher pytorch



# cd tools
# # python linear_probing_kfold.py --model_root "../work_dirs/selfsup/byol/byol-gaze_resnet50_1xb64-300e_MammoData_everlasting/" --model_name "epoch_200.pth" --lr 3e-3 --wd 1e-5
# # # python feature_eval_kfold.py --model_root "../work_dirs/selfsup/byol/byol-gaze_resnet50_1xb64-300e_MammoData_everlasting/" --model_name "epoch_200.pth"
# # python linear_probing_kfold.py --model_root "../work_dirs/selfsup/byol/byol-cluster_resnet50-400e_Mammo_p0.5-dual0.95/" --model_name "epoch_200.pth" --lr 4e-3 --wd 1e-5
# # python linear_probing_kfold.py --model_root "../work_dirs/selfsup/byol/byol-gaze_resnet50_Mammo_p0.5/" --model_name "epoch_300.pth"
# python feature_eval_kfold.py --model_root "../work_dirs/selfsup/byol/-sup_resnet101_4xb32-200e_Mammo/" --model_name "epoch_200.pth" --arch "resnet101"
# python feature_eval_kfold.py --model_root "../work_dirs/selfsup/simsiam-cluster_resnet50-200e_Mammo/" --model_name "epoch_200.pth" --lr 2e-5
# # python feature_eval_kfold.py --model_root "../work_dirs/selfsup/byol/byol-cluster_resnet50-400e_Mammo_p0.5-dual0.95/" --model_name "epoch_300.pth" --lr 8e-6
# # python feature_eval_kfold.py --model_root "../work_dirs/selfsup/byol/byol-cluster_resnet50-400e_Mammo_p0.5-dual0.95/" --model_name "epoch_300.pth" --lr 6e-6
# # python feature_eval_kfold.py --model_root "../work_dirs/selfsup/byol/byol-cluster_resnet50-400e_Mammo_p0.5-dual0.95/" --model_name "epoch_300.pth" --lr 8e-5
# # python feature_eval_kfold.py --model_root "../work_dirs/selfsup/byol/byol-cluster_resnet50-400e_Mammo_p0.5-dual0.95/" --model_name "epoch_300.pth" --lr 1e-4
# # python linear_probing_kfold.py --model_root "../work_dirs/selfsup/byol/byol-cluster_resnet50-300e_Mammo_hlr_dual0.99/" --model_name "epoch_100.pth" --lr 4e-3 --wd 1e-5
# # python feature_eval_kfold.py --model_root "../work_dirs/selfsup/byol/byol-cluster_resnet50-300e_Mammo_hlr_dual0.99/" --model_name "epoch_300.pth" --lr 8e-5 --wd 1e-3

