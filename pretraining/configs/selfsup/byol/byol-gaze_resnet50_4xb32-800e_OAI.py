_base_ = [
    # '../_base_/models/byol.py',
    '../_base_/datasets/OAI.py',
    # '../_base_/schedules/lars_coslr-200e_in1k.py',
    '../_base_/schedules/adamw_coslr-200e_in1k.py',
    '../_base_/default_runtime.py',
]
weights='0.1-0.1-0.1-0.1-0.6'
threshold=0.95
p=0.5
relation_file="../data/OAI/relation_match.npy"
data = dict(
    samples_per_gpu=32,  # total 32*8
    workers_per_gpu=4,
    drop_last=False,
    )
prefetch=False
# model settings
model = dict(
    type='BYOL_gaze',
    weights=weights,
    threshold=threshold,
    p=p,
    base_momentum=0.99,
    backbone=dict(
        type='ResNet',
        depth=50,
        in_channels=3,
        out_indices=[4],  # 0: conv-1, x: stage-x
        norm_cfg=dict(type='SyncBN'),
        init_cfg=dict(
            type='Pretrained',
            checkpoint='../preTrain/resnet50-19c8e357.pth',
                ),
    ),
    neck=dict(
        type='NonLinearNeck',
        in_channels=2048,
        hid_channels=4096,
        out_channels=256,
        num_layers=2,
        with_bias=True,
        with_last_bn=False,
        with_avg_pool=True),
    head=dict(
        type='LatentGazePredictHead',
        predictor=dict(
            type='NonLinearNeck',
            in_channels=256,
            hid_channels=4096,
            out_channels=256,
            num_layers=2,
            with_bias=True,
            with_last_bn=False,
            with_avg_pool=False)))

# additional hooks
# interval for accumulate gradient, total 8*32*16(interval)=4096
update_interval = 4
custom_hooks = [
    dict(type='BYOLHook', end_momentum=1., update_interval=update_interval)
]

# optimizer
optimizer = dict(type='AdamW', lr=8e-4, betas=(0.9, 0.95), weight_decay=1e-6)
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
optimizer_config = dict(update_interval=update_interval)
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0.,
    warmup='linear',
    warmup_iters=10,
    warmup_ratio=1e-4,  # cannot be 0
    warmup_by_epoch=True)
runner=dict(max_epochs=800)
# runtime settings
# the max_keep_ckpts controls the max number of ckpt file in your work_dirs
# if it is 3, when CheckpointHook (in mmcv) saves the 4th ckpt
# it will remove the oldest one to keep the number of total ckpts as 3
checkpoint_config = dict(interval=800, max_keep_ckpts=1)
log_config = dict(
    interval=5,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook'),
    ])
