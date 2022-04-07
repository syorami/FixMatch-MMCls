_base_ = [f'./rand_aug.py']

# dataset settings
dataset_type = 'CIFAR10'
img_norm_cfg = dict(
    mean=[125.307, 122.961, 113.8575],
    std=[51.5865, 50.847, 51.255],
    to_rgb=False)

meta_keys=('filename', 'ori_filename', 'ori_shape',
           'img_shape', 'flip', 'flip_direction',
           'img_norm_cfg', 'tag')

weak_pipeline = [
    dict(type='RandomCrop', size=32, padding=4),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type="ExtraAttrs", tag="sup"),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'], meta_keys=meta_keys)
]

strong_pipeline = [
    dict(type='RandomCrop', size=32, padding=4),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(
        type='RandAugment',
        policies={{_base_.rand_increasing_policies}},
        num_policies=2,
        total_level=10,
        magnitude_level=9,
        magnitude_std=0.5,
        hparams=dict(
            pad_val=[round(x) for x in img_norm_cfg['mean'][::-1]],
            interpolation='bicubic')),
    dict(type='Normalize', **img_norm_cfg),
    dict(type="ExtraAttrs", tag="unsup"),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'], meta_keys=meta_keys)
]
test_pipeline = [
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]

samples_per_gpu = 64
batch_nums = (8, 56)
data = dict(
    samples_per_gpu=samples_per_gpu,
    workers_per_gpu=2,
    sampler=dict(type='DistributedBalanceSampler',
                 samples_per_gpu=samples_per_gpu,
                 batch_nums=batch_nums,
                 len_type='max'),
    train=dict(
        type='SemiDataset',
        sup=dict(
            type='SubsetDataset',
            indices='data/cifar10_4000/labeled_idx.pkl',
            dataset=dict(
                type=dataset_type,
                data_prefix='data/cifar10',
                pipeline=weak_pipeline)),
        unsup=dict(
            type='SubsetDataset',
            indices='data/cifar10_4000/unlabeled_idx.pkl',
            dataset=dict(
                type=dataset_type,
                data_prefix='data/cifar10',
                pipeline=strong_pipeline))),
    val=dict(
        type=dataset_type,
        data_prefix='data/cifar10',
        pipeline=test_pipeline,
        test_mode=True),
    test=dict(
        type=dataset_type,
        data_prefix='data/cifar10',
        pipeline=test_pipeline,
        test_mode=True))

# evaluation = dict(interval=1024, metric='accuracy')

evaluation = dict(type="SubModulesDistEvalHook",
                  interval=1024,
                  metric='accuracy')