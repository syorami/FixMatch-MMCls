# optimizer
optimizer = dict(type='SGD', lr=0.03, momentum=0.9, weight_decay=5e-4)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='cosine')
runner = dict(type='IterBasedRunner', max_iters=2**20)
