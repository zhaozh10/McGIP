
_base_ = [
    '../_base_/models/simclr.py',
    '../_base_/datasets/Mammo.py',
    '../_base_/schedules/adamw_coslr-200e_in1k.py',
    '../_base_/default_runtime.py',
]
weights='0.1-0.1-0.1-0.1-0.6'
threshold=0.7
# model settings
model = dict(
    type='SimCLR_gaze',
    weights=weights,
    threshold=threshold,
    backbone=dict(
        # Here we add init_cfg for backbone to load pretrained weights
        init_cfg=dict(
                type='Pretrained',
                # checkpoint='https://download.pytorch.org/models/resnet50-19c8e357.pth',
                checkpoint='https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb32_in1k_20210831-ea4938fc.pth',
                prefix='backbone',
                ),
        type='ResNet',
        depth=50,
        in_channels=3,
        out_indices=[4],  # 0: conv-1, x: stage-x
        norm_cfg=dict(type='SyncBN'),
        zero_init_residual=True),
    neck=dict(
        type='NonLinearNeck',  # SimCLR non-linear neck
        in_channels=2048,
        hid_channels=2048,
        out_channels=128,
        num_layers=2,
        with_avg_pool=True),
    head=dict(type='GazeHead', temperature=0.07))


# optimizer
optimizer = dict(type='AdamW', lr=2e-5, betas=(0.9, 0.95), weight_decay=1e-6)
# optimizer = dict(
#     type='LARS',
#     lr=1e-3,
#     momentum=0.9,
#     weight_decay=1e-6,
#     paramwise_options={
#         '(bn|gn)(\\d+)?.(weight|bias)':
#         dict(weight_decay=0., lars_exclude=True),
#         'bias': dict(weight_decay=0., lars_exclude=True)
#     })
# optimizer_config = dict(update_interval=update_interval)
runner=dict(max_epochs=270)
# runtime settings
# the max_keep_ckpts controls the max number of ckpt file in your work_dirs
# if it is 3, when CheckpointHook (in mmcv) saves the 4th ckpt
# it will remove the oldest one to keep the number of total ckpts as 3
checkpoint_config = dict(interval=30, max_keep_ckpts=3)




