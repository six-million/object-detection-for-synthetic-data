_base_ = "./coco_detection.py"

# dataset settings
dataset_type = 'CocoDataset'
data_root = '../open/'

backend_args = None

img_norm_cfg = dict(
    mean=[105.83, 110.62, 111.49],
    std=[46.50, 54.21, 59.66],
    to_rgb=True
)

albu_train_transforms = [
    dict(
        type='OneOf',
        transforms=[
            dict(
                type='RandomSizedBBoxSafeCrop',
                height=512,
                width=512,
                p=1.0,
            ),
            dict(
                type='CropNonEmptyMaskIfExists',
                height=512,
                width=512,
                p=1.0,
            ),
        ],
        p=1.0),
    # dict(
    #     type='ShiftScaleRotate',
    #     shift_limit=0.0625,
    #     scale_limit=0.0,
    #     rotate_limit=0,
    #     interpolation=1,
    #     p=0.5),
    dict(
        type='RandomBrightnessContrast',
        brightness_limit=[0.1, 0.3],
        contrast_limit=[0.1, 0.3],
        p=0.2),
    dict(
        type='OneOf',
        transforms=[
            dict(
                type='RGBShift',
                r_shift_limit=10,
                g_shift_limit=10,
                b_shift_limit=10,
                p=1.0),
            dict(
                type='HueSaturationValue',
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=1.0)
        ],
        p=0.1),
    # dict(type='JpegCompression', quality_lower=85, quality_upper=95, p=0.2),
    # dict(type='ChannelShuffle', p=0.1),
    dict(
        type='OneOf',
        transforms=[
            dict(type='Blur', blur_limit=3, p=1.0),
            dict(type='MedianBlur', blur_limit=3, p=1.0)
        ],
        p=0.1),
]

train_pipeline = [
    # dict(type="MinIoURandomCrop"),
    # dict(type="PhotoMetricDistortion"), # 밝기, 대조, HSV shift, saturation, hue, RGB shift, 등등 p=0.5로 랜덤으로 적용
    # dict(type="Corrupt", corruption="motion_blur"), # noise, blur 를 corruption으로 지정
    dict(
        type='Albu',
        transforms=albu_train_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_labels'],
            min_visibility=0.0,
            filter_lost_elements=True),
        keymap={
            'img': 'image',
            'gt_masks': 'masks',
            'gt_bboxes': 'bboxes'
        },
        update_pad_shape=False,
        skip_img_without_anno=True),
    dict(type="Normalize", **img_norm_cfg),
    dict(type='PackDetInputs')
]
data = dict(train=dict(pipeline=train_pipeline))