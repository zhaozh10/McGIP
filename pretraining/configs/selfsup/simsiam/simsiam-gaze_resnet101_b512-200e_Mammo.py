_base_ = [
    # '../_base_/models/simsiam.py',
    '../_base_/datasets/Mammo.py',
    # '../_base_/schedules/adamw_coslr-200e_in1k.py',
    # '../_base_/schedules/sgd_coslr-200e_in1k.py',
    # '../_base_/schedules/sgd_coslr-200e_in1k.py',
'../_base_/schedules/adamw_coslr-200e_in1k.py',
    '../_base_/default_runtime.py',
]
weights='0.1-0.1-0.1-0.1-0.6'
threshold=0.7

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
    type='SimSiam_gaze',
    weights=weights,
    threshold=threshold,
    backbone=dict(
        type='ResNet',
        depth=101,
        in_channels=3,
        out_indices=[4],  # 0: conv-1, x: stage-x
        norm_cfg=dict(type='SyncBN'),
        init_cfg=dict(
                type='Pretrained',
                checkpoint='https://download.openmmlab.com/mmclassification/v0/resnet/resnet101_8xb32_in1k_20210831-539c63f8.pth',
                prefix='backbone',
                ),
        # zero_init_residual=True
    ),
    neck=dict(
        type='NonLinearNeck',
        in_channels=2048,
        hid_channels=4096,
        out_channels=256,
        num_layers=2,
        with_last_bn_affine=False,
        with_avg_pool=True),
    head=dict(
        type='LatentGazePredictHead',
        predictor=dict(
            type='NonLinearNeck',
            in_channels=256,
            hid_channels=4096,
            out_channels=256,
            with_avg_pool=False,
            with_last_bn=False,
            with_bias=True)))




update_interval = 16
custom_hooks = [
    dict(type='BYOLHook', end_momentum=1., update_interval=update_interval)
]

# optimizer
optimizer = dict(type='AdamW', lr=2e-5, betas=(0.9, 0.95), weight_decay=1e-6)

optimizer_config = dict(update_interval=update_interval)

runner=dict(max_epochs=200)
checkpoint_config = dict(interval=100, max_keep_ckpts=2)