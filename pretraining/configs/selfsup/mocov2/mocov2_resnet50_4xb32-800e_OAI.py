_base_ = [
    '../_base_/datasets/OAI.py',
    # '../_base_/schedules/lars_coslr-200e_in1k.py',
    '../_base_/schedules/adamw_coslr-200e_in1k.py',
    '../_base_/default_runtime.py',
]
data = dict(
    samples_per_gpu=32,  # total 32*8
    workers_per_gpu=4,
    drop_last=True,
    )
model = dict(
    type='MoCo',
    queue_len=512,
    feat_dim=256,
    momentum=0.999,
    backbone=dict(
    # Here we add init_cfg for backbone to load pretrained weights
        init_cfg=dict(
                type='Pretrained',
                checkpoint='../preTrain/resnet50-19c8e357.pth',
                # type='Pretrained',
                # checkpoint='https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb32_in1k_20210831-ea4938fc.pth',
                # prefix='backbone',
                ),
        type='ResNet',
        depth=50,
        in_channels=3,
        out_indices=[4],  # 0: conv-1, x: stage-x
        norm_cfg=dict(type='BN')),
    neck=dict(
        type='MoCoV2Neck',
        in_channels=2048,
        hid_channels=2048,
        out_channels=256,
        with_avg_pool=True),
    head=dict(type='ContrastiveHead', temperature=0.2))

# optimizer
optimizer = dict(type='AdamW', lr=8e-4, betas=(0.9, 0.95), weight_decay=1e-6)
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
