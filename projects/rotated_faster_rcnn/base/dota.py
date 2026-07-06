# dataset settings
dataset_type = 'DOTADataset'
data_root = '/home/leo/dataset/drawings/toy_ab_af_c_lc_tc_d_an_cn_em/dota/'

# 自定义类别（与 projects/rotated_rtdetr/configs_custom/dota_ms.py 一致）
classes = (
    "angelSteelBack",
    "angelSteelFront",
    "clamp",
    "LConnection",
    "TConnection",
    "dimension",
    "angelSteelNumber",
    "clampNumber",
    "endMark")
palette = [(189, 183, 107), (0, 255, 0), (255, 0, 0),
           (138, 43, 226), (255, 128, 0), (255, 0, 255),
           (0, 255, 255), (128, 0, 255), (255, 255, 0)]
metainfo = dict(classes=classes, palette=palette)

backend_args = None

train_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args=backend_args),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(type='mmdet.Resize', scale=(640, 640), keep_ratio=True),
    dict(
        type='mmdet.RandomFlip',
        prob=0.75,
        direction=['horizontal', 'vertical', 'diagonal']),
    dict(type='mmdet.PackDetInputs')
]
val_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args=backend_args),
    dict(type='mmdet.Resize', scale=(640, 640), keep_ratio=True),
    # avoid bboxes being resized
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
test_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args=backend_args),
    dict(type='mmdet.Resize', scale=(640, 640), keep_ratio=True),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
train_dataloader = dict(
    batch_size=4,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=None,
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
    num_workers=2,
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

val_evaluator = dict(type='DOTAMetric', 
                     metric='mAP',
                     iou_thrs=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95])
# test_evaluator = val_evaluator

# inference on test dataset and format the output results
# for submission. Note: the test set has no annotation.
test_dataloader = dict(
    batch_size=4,
    num_workers=2,
    persistent_workers=True,
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
    outfile_prefix='./work_dirs/dota/Task1')
