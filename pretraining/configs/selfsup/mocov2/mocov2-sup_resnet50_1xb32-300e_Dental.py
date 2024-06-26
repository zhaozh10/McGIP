_base_ = [
    # '../_base_/models/mocov2.py',
    '../_base_/datasets/Dental.py',
    # '../_base_/schedules/lars_coslr-200e_in1k.py',
    '../_base_/schedules/adamw_coslr-200e_in1k.py',
    '../_base_/default_runtime.py',
]

data = dict(
    samples_per_gpu=16,  # total 32*8
    workers_per_gpu=4,
    drop_last=True,
    )
# model settings
model = dict(
    type='MoCo_sup',
    queue_len=320,
    base_momentum=0.99,
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
    ),
    neck=dict(
        type='MoCoV2Neck',
        in_channels=2048,
        hid_channels=2048,
        out_channels=256,
        with_avg_pool=True),
    head=dict(type='MoCoGazeHead', temperature=0.2)
            )


# optimizer
optimizer = dict(type='AdamW', lr=2e-5, betas=(0.9, 0.95), weight_decay=1e-6)
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0.,
    warmup='linear',
    warmup_iters=10,
    warmup_ratio=1e-4,  # cannot be 0
    warmup_by_epoch=True)
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

runner=dict(max_epochs=300)
# runtime settings
# the max_keep_ckpts controls the max number of ckpt file in your work_dirs
# if it is 3, when CheckpointHook (in mmcv) saves the 4th ckpt
# it will remove the oldest one to keep the number of total ckpts as 3
checkpoint_config = dict(interval=300, max_keep_ckpts=1)
log_config = dict(
    interval=5,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook'),
    ])
