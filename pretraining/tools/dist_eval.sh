#!/usr/bin/env bash

for i in {1..10}
do
#    python feature_eval_kfold.py --seed ${i} --data_prefix "../ttest/byol_gaze" --model_root "../work_dirs/selfsup/byol/byol-gaze_resnet50_1xb64-300e_MammoData_everlasting/" --model_name "epoch_100.pth"
   python feature_eval_kfold.py --seed ${i} --data_prefix "../ttest/simsiam/gaze/simsiam_gaze" --model_root "../work_dirs/selfsup/simsiam/simsiam-gaze_resnet50_4xb128-200e_Mammo/" --model_name "epoch_200.pth"
   python feature_eval_kfold.py --seed ${i} --data_prefix "../ttest/simsiam/wo_gaze/simsiam" --model_root "../work_dirs/selfsup/simsiam/simsiam_resnet50_1xb64-coslr-200e_Mammo/" --model_name "epoch_200.pth"
done
# python feature_eval_kfold.py --model_root --seed 0 --model_root "../work_dirs/selfsup/byol/byol-gaze_resnet50_1xb64-300e_MammoData_everlasting/" --model_name "epoch_100.pth"
# python feature_eval_kfold.py --model_root "../work_dirs/selfsup/byol/byol-cluster_resnet50-400e_Mammo_p0.5-dual0.95/" --model_name "epoch_300.pth" --lr 8e-6
# python feature_eval_kfold.py --model_root "../work_dirs/selfsup/byol/byol-cluster_resnet50-400e_Mammo_p0.5-dual0.95/" --model_name "epoch_300.pth" --lr 6e-6
# python feature_eval_kfold.py --model_root "../work_dirs/selfsup/byol/byol-cluster_resnet50-400e_Mammo_p0.5-dual0.95/" --model_name "epoch_300.pth" --lr 8e-5
# python feature_eval_kfold.py --model_root "../work_dirs/selfsup/byol/byol-cluster_resnet50-400e_Mammo_p0.5-dual0.95/" --model_name "epoch_300.pth" --lr 1e-4