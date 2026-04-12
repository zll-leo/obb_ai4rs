# dataset settings
dataset_type = 'DOTADataset'
data_root = '/home/leo/dataset/drawings/ab_af_c_c_d_anno/dota'

# 自定义类别
classes = (
    "angelSteelBack",
    "angelSteelFront",
    "clamp",
    "LConnection",
    "TConnection",
    "dimension")
# 为每个类别生成颜色（palette）
palette = [(189, 183, 107), (0, 255, 0), (255, 0, 0),
           (138, 43, 226), (255, 128, 0), (255, 0, 255)]

metainfo = dict(classes=classes, palette=palette)

backend_args = None

train_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args=backend_args),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(type='mmdet.Resize', scale=(1024, 1024), keep_ratio=True),
    dict(
        type='mmdet.RandomFlip',
        prob=0.75,
        direction=['horizontal', 'vertical', 'diagonal']),
    dict(
        type='RandomRotate',
        prob=0.5,
        angle_range=180,
        rect_obj_labels=[]),  # 自定义数据集无需矩形对象标签
    dict(
        type='mmdet.Pad', size=(1024, 1024),
        pad_val=dict(img=(114, 114, 114))),
    dict(type='mmdet.PackDetInputs')
]
val_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args=backend_args),
    dict(type='mmdet.Resize', scale=(1024, 1024), keep_ratio=True),
    # avoid bboxes being resized
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(
        type='mmdet.Pad', size=(1024, 1024),
        pad_val=dict(img=(114, 114, 114))),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
test_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args=backend_args),
    dict(type='mmdet.Resize', scale=(1024, 1024), keep_ratio=True),
    dict(
        type='mmdet.Pad', size=(1024, 1024),
        pad_val=dict(img=(114, 114, 114))),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=None,
    pin_memory=False,
    dataset=dict(
        type=dataset_type,
        metainfo=metainfo,
        data_root=data_root,
        ann_file='labels/train/',
        data_prefix=dict(img_path='images/train/'),
        filter_cfg=dict(filter_empty_gt=True),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        metainfo=metainfo,
        data_root=data_root,
        ann_file='labels/val/',
        data_prefix=dict(img_path='images/val/'),
        test_mode=True,
        pipeline=val_pipeline))
# test_dataloader = val_dataloader

val_evaluator = dict(type='DOTAMetric', metric='mAP')
# test_evaluator = val_evaluator

# inference on test dataset and format the output results
# for submission. Note: the test set has no annotation.
test_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        metainfo=metainfo,
        data_root=data_root,
        data_prefix=dict(img_path='test/images/'),
        test_mode=True,
        pipeline=test_pipeline))
test_evaluator = dict(
    type='DOTAMetric',
    format_only=True,
    merge_patches=True,
    outfile_prefix='./work_dirs/Task1')

# custom hooks
custom_hooks = [
    dict(type='mmdet.NumClassCheckHook')
]

# runtime settings
default_scope = 'mmrotate'

default_hooks = dict(
    timer=dict(type='IterTimerHook'),                           # 记录每次迭代耗时
    logger=dict(type='LoggerHook', interval=50),                # 每50次迭代打印日志
    param_scheduler=dict(type='ParamSchedulerHook'),            # 学习率调度
    checkpoint=dict(type='CheckpointHook', interval=3, max_keep_ckpts=3),   # 保存模型，最多保留3个
    sampler_seed=dict(type='DistSamplerSeedHook'),              # 分布式采样随机种子 
    visualization=dict(type='mmdet.DetVisualizationHook'))      # 可视化预测结果 

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='RotLocalVisualizer', vis_backends=vis_backends, name='visualizer')
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)

log_level = 'INFO'
load_from = None
resume = False