# model settings
arch = dict(
    type='ImageClassifier',
    backbone=dict(
        type='WideResNetCIFAR',
        depth=28,
        widen_factor=2,
        drop_rate=0.),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=10,
        in_channels=128,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    ))
