_base_ = [
    '../_base_/models/mocov2.py',
    '../_base_/datasets/Mammo.py',
    '../_base_/schedules/lars_coslr-200e_in1k.py',
    '../_base_/default_runtime.py',
]
threshold=0.95
relation= '../data/INBreast/correlation.npy'
# model settings
model = dict(
    type='MoCo_McGIP',
    threshold=threshold,
    queue_len=1280,
    feat_dim=256,
    momentum=0.999,
    backbone=dict(
    # Here we add init_cfg for backbone to load pretrained weights
        init_cfg=dict(
                type='Pretrained',
                checkpoint='https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb32_in1k_20210831-ea4938fc.pth',
                prefix='backbone',
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
    head=dict(type='MoCoGazeHead', temperature=0.2))
# optimizer
optimizer = dict(
    type='LARS',
    # lr=basic lr*batchsize*n_GPU/(64*1)
    lr=2e-3*(64*1)/(64*1),
    momentum=0.9,
    weight_decay=1e-2,
    paramwise_options={
        '(bn|gn)(\\d+)?.(weight|bias)':
        dict(weight_decay=0., lars_exclude=True),
        'bias': dict(weight_decay=0., lars_exclude=True)
    })

# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0.,
    warmup='linear',
    warmup_iters=10,
    warmup_ratio=1e-4,
    warmup_by_epoch=True)
runner=dict(max_epochs=300)



# runtime settings
# the max_keep_ckpts controls the max number of ckpt file in your work_dirs
# if it is 3, when CheckpointHook (in mmcv) saves the 4th ckpt
# it will remove the oldest one to keep the number of total ckpts as 3
checkpoint_config = dict(interval=30, max_keep_ckpts=6)