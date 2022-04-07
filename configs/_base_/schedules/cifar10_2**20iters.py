# optimizer
optimizer = dict(type='SGD', lr=0.03, momentum=0.9, weight_decay=1e-4)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0,
    warmup='linear',
    warmup_iters=5000,
    warmup_ratio=0.25)
runner = dict(type='IterBasedRunner', max_iters=2**20)
