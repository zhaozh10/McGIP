data_prefix=''
ann_file='sort_namelist.txt'
dataset_type = 'MammoCLR'
data_source='MammoDataSource'
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_pipeline = [
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
