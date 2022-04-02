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