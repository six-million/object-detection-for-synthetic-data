# import datetime
# from pytz import timezone
# now = datetime.datetime.now(timezone('Asia/Seoul')).strftime('_%y%m%d_%H%M%S')
# checkpoint_config = dict(interval=1)
# # yapf:disable
# log_config = dict(
#     interval=50,
#     hooks=[
#         dict(type='TextLoggerHook'),
#          dict(type='WandbLoggerHook',interval=1000,
#             init_kwargs=dict(
#                 project="object_detection",
#                 entity ="wooyeolbaek",
#                 name = "base"+now
#             ),)
#     ])
# # yapf:enable
# custom_hooks = [dict(type='NumClassCheckHook')]

# dist_params = dict(backend='nccl')
##################################################################################
default_scope = 'mmdet'

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'))

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)

log_level = 'INFO'
load_from = None
resume = False
