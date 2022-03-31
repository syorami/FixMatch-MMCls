_base_ = [
    '../_base_/models/wide-resnet50.py',
    '../_base_/datasets/semi_cifar10_bs16.py',
    '../_base_/schedules/cifar10_2**20iters.py',
    '../_base_/default_runtime.py'
]