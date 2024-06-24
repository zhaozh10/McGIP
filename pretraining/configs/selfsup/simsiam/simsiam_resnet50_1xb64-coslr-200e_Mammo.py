_base_ = [
    # '../_base_/models/simsiam.py',
    '../_base_/datasets/Mammo.py',
    # '../_base_/schedules/adamw_coslr-200e_in1k.py',
    '../_base_/schedules/sgd_coslr-200e_in1k.py',
    # '../_base_/schedules/sgd_coslr-200e_in1k.py',
    '../_base_/default_runtime.py',
]

model = dict(
    type='SimSiam',
    backbone=dict(
        type='ResNet',
        depth=50,
        in_channels=3,
        out_indices=[4],  # 0: conv-1, x: stage-x
        norm_cfg=dict(type='SyncBN'),
        init_cfg=dict(
                type='Pretrained',
                checkpoint='https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb32_in1k_20210831-ea4938fc.pth',
                prefix='backbone',
                ),
        # zero_init_residual=True
    ),
    neck=dict(
        type='NonLinearNeck',
        in_channels=2048,
        hid_channels=2048,
        out_channels=2048,
        num_layers=3,
        with_last_bn_affine=False,
        with_avg_pool=True),
    head=dict(
        type='LatentPredictHead',
        predictor=dict(
            type='NonLinearNeck',
            in_channels=2048,
            hid_channels=512,
            out_channels=2048,
            with_avg_pool=False,
            with_last_bn=False,
            with_last_bias=True)))


lr = 0.05

# additional hooks
custom_hooks = [
    dict(type='SimSiamHook', priority='HIGH', fix_pred_lr=True, lr=lr)
]

# optimizer
optimizer = dict(lr=lr, paramwise_options={'predictor': dict(fix_lr=True)})

# runtime settings
# the max_keep_ckpts controls the max number of ckpt file in your work_dirs
# if it is 3, when CheckpointHook (in mmcv) saves the 4th ckpt
# it will remove the oldest one to keep the number of total ckpts as 3
checkpoint_config = dict(interval=50, max_keep_ckpts=3)
runner=dict(max_epochs=200)

