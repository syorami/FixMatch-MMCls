# checkpoint saving
checkpoint_config = dict(interval=1024)
# yapf:disable
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]

find_unused_parameters = True

custom_hooks = [
    dict(type="MeanTeacher", momentum=0.999, interval=1, warm_up=0),
]
