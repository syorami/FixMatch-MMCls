_base_ = [
    '../_base_/models/wide-resnet28x2-cifar10.py',
    '../_base_/datasets/semi_cifar10_bs64.py',
    '../_base_/schedules/cifar10_2**20iters.py',
    '../_base_/default_runtime.py'
]

model = dict(
    type='FixMatch',
    student={{_base_.arch}},
    teacher={{_base_.arch}},
    train_cfg=dict(
        temperature=1,
        score_thr=0.95,
        unsup_weight=1
    ),
    test_cfg=dict(
        inference_on=('student', 'teacher')
    )
)

data = dict(
    train=dict(
        sup=dict(indices='data/cifar10_40/labeled_idx.pkl'),
        unsup=dict(indices='data/cifar10_40/unlabeled_idx.pkl')
    )
)

custom_hooks = [
    dict(type="MeanTeacher", momentum=0.999, interval=1, warm_up=5000),
]

fp16 = dict(loss_scale='dynamic')
