
_base_ = [
    '../_base_/models/simclr.py',
    # '../_base_/datasets/imagenet_simclr.py',
    '../_base_/schedules/lars_coslr-200e_in1k.py',
    '../_base_/default_runtime.py',
]

# model settings
model = dict(
    type='SimCLR',
    backbone=dict(
        # Here we add init_cfg for backbone to load pretrained weights
        init_cfg=dict(
                type='Pretrained',
                checkpoint='https://download.openmmlab.com/mmclassification/v0/resnet/resnet101_8xb32_in1k_20210831-539c63f8.pth',
                prefix='backbone',
                ),
        type='ResNet',
        depth=101,
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
    head=dict(type='ContrastiveHead', temperature=0.07))



data_prefix='/home/zihao/mmself/'
ann_file='/home/zihao/mmself/namelist.txt'
dataset_type = 'MammoCLR'
data_source='MammoDataSource'
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_pipeline = [
    # dict(type='Resize', size=(224,224),interpolation=InterpolationMode.BILINEAR),
    # dict(type='RandomResizedCrop', size=224),
    dict(type='RandomHorizontalFlip'),
    dict(
        type='RandomAppliedTrans',
        transforms=[
            dict(
                type='ColorJitter',
                brightness=0.8,
                contrast=0.8,
                saturation=0.8,
                hue=0.2)
        ],
        p=0.8),
    dict(type='GaussianBlur', sigma_min=0.1, sigma_max=2.0, p=0.5),
]
prefetch = False
if not prefetch:
    train_pipeline.extend(
        [
        # dict(type='ToPILImage'),
        dict(type='ToTensor'),
        dict(type='Normalize', **img_norm_cfg)
        ])
# dataset summary
data = dict(
    samples_per_gpu=16,  # total 32*8
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_source=dict(
            type=data_source,
            data_prefix=data_prefix,
            ann_file=ann_file,
        ),
        num_views=[2],
        pipelines=[train_pipeline],
        prefetch=prefetch,
    ))




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
runner=dict(max_epochs=200)
# runtime settings
# the max_keep_ckpts controls the max number of ckpt file in your work_dirs
# if it is 3, when CheckpointHook (in mmcv) saves the 4th ckpt
# it will remove the oldest one to keep the number of total ckpts as 3
checkpoint_config = dict(interval=20, max_keep_ckpts=3)




