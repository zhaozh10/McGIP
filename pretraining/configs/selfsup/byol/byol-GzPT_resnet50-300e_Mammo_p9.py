_base_ = [
    # '../_base_/models/mocov2.py',
    '../_base_/datasets/Mammo_heat.py',
    # '../_base_/schedules/lars_coslr-200e_in1k.py',
    '../_base_/schedules/adamw_coslr-200e_in1k.py',
    '../_base_/default_runtime.py',
]
threshold = 0.95
p = 0.9
# relation_combine_heat
relation= '../data/INBreast/correlation_tima.npy'
# relation='relation_cluster_heat.npy'
# model settings
data = dict(
    samples_per_gpu=32,  # total 32*8
    workers_per_gpu=2,
    train=dict(
        type={{_base_.dataset_type}},
        data_source=dict(
            type={{_base_.data_source}},
            data_prefix={{_base_.data_prefix}},
            ann_file={{_base_.ann_file}},
        ),
        num_views=[2],
        pipelines=[{{_base_.train_pipeline}}],
        prefetch={{_base_.prefetch}},
    ))
model = dict(
    type='BYOL_heat',
    threshold=threshold,
    p=p,
    base_momentum=0.99,
    relation=relation,
    # base_momentum=0,
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
# interval for accumulate gradient, total 32*16(interval)=512
update_interval = 16
custom_hooks = [
    dict(type='BYOLHook', end_momentum=1., update_interval=update_interval)
]

# optimizer
optimizer = dict(type='AdamW', lr=2e-5, betas=(0.9, 0.95), weight_decay=1e-6)
# optimizer = dict(type='AdamW', lr=6e-5, betas=(0.9, 0.95), weight_decay=1e-6)
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
runner=dict(max_epochs=200)
# runtime settings
# the max_keep_ckpts controls the max number of ckpt file in your work_dirs
# if it is 3, when CheckpointHook (in mmcv) saves the 4th ckpt
# it will remove the oldest one to keep the number of total ckpts as 3
checkpoint_config = dict(interval=100, max_keep_ckpts=2)