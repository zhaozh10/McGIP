data_prefix='../data/OAI/img/'
ann_file='../data/OAI/preTrain.json'
dataset_type = 'OAIset'
data_source='OAIsource'
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_pipeline = [
    # dict(type='Resize', size=224),
    dict(type='RandomResizedCrop', size=224, interpolation=3),
    # dict(type='Resize', size=224),
    dict(type='RandomHorizontalFlip',p=0.5),
    # dict(
    #     type='RandomAppliedTrans',
    #     transforms=[
    #         dict(
    #             type='ColorJitter',
    #             brightness=0.8,
    #             contrast=0.8,
    #             saturation=0.8,
    #             hue=0.2)
    #     ],
    #     p=0.8),
    dict(type='GaussianBlur', sigma_min=0.1, sigma_max=2.0, p=0.5),
    dict(type='Solarization', p=0.2),
]
prefetch = False
if not prefetch:
    train_pipeline.extend(
        [
        dict(type='ToTensor'),
        dict(type='Normalize', **img_norm_cfg)
        ])
# dataset summary
data = dict(
    samples_per_gpu=32,  # total 32*8
    workers_per_gpu=4,
    drop_last=False,
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
