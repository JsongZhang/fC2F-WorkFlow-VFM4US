# configs/upernet_vit_config.py

_base_ = ['./_base_/models/upernet.py', './_base_/datasets/cityscapes.py', './_base_/default_runtime.py']

model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='VisionTransformer',
        embed_dims=768,
        num_heads=12,
        num_layers=12,
        patch_size=16,
        pretrained='path_to_your_pretrained_vit_weights.pth',  # 预训练权重
    ),
    decode_head=dict(
        type='UPerHead',
        in_channels=[768],
        channels=512,
        num_classes=19,  # 修改为你的数据集类别数
        dropout_ratio=0.1,
        num_convs=4,
        align_corners=False,
    ),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=768,
        channels=256,
        num_classes=19,
        dropout_ratio=0.1,
    ),
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)

# 数据集配置
data = dict(
    train=dict(
        type='CustomDataset',  # 自定义数据集
        data_root='path_to_your_data',  # 数据路径
        img_dir='train/images',
        ann_dir='train/annotations',
    ),
    val=dict(
        type='CustomDataset',
        data_root='path_to_your_data',
        img_dir='val/images',
        ann_dir='val/annotations',
    ),
)

# 其他配置项（学习率、优化器等）
optimizer = dict(type='AdamW', lr=1e-4, weight_decay=0.01)
