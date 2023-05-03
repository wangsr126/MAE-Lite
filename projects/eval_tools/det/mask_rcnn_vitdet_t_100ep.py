import detectron2.data.transforms as T
from detectron2 import model_zoo
from detectron2.config import LazyCall as L
from functools import partial
from fvcore.common.param_scheduler import MultiStepParamScheduler
from detectron2.solver import WarmupParamScheduler
from detectron2.modeling.backbone.vit import get_vit_lr_decay_rate

# image_size = 768
image_size = 640
total_batch_size = 64
total_epoch = 100

# Data using LSJ
dataloader = model_zoo.get_config("common/data/coco.py").dataloader
dataloader.train.mapper.augmentations = [
    L(T.RandomFlip)(horizontal=True),  # flip first
    L(T.ResizeScale)(
        min_scale=0.1, max_scale=2.0, target_height=image_size, target_width=image_size
    ),
    L(T.FixedSizeCrop)(crop_size=(image_size, image_size), pad=False),
]
dataloader.train.mapper.image_format = "RGB"
dataloader.train.total_batch_size = total_batch_size
# recompute boxes due to cropping
dataloader.train.mapper.recompute_boxes = True
dataloader.test.mapper.augmentations = [
    L(T.ResizeShortestEdge)(short_edge_length=image_size, max_size=image_size),
]

model = model_zoo.get_config("common/models/mask_rcnn_vitdet.py").model
model.backbone.net.embed_dim = 192
model.backbone.net.num_heads = 12
model.backbone.net.drop_path_rate = 0.0
model.backbone.net.img_size = image_size
model.backbone.net.use_act_checkpoint = True
model.backbone.square_pad = image_size

# Initialization and trainer settings
train = model_zoo.get_config("common/train.py").train
train.amp.enabled = True
train.ddp.fp16_compression = True
train.init_checkpoint = "checkpoints/deit_tiny_300e.pth?matching_heuristics=True"
train.output_dir = "output/ViTDet/mask_rcnn_vitdet_t_100ep"

# Schedule
# 100 ep = 184375 iters * 64 images/iter / 118000 images/ep
_ratio = 64 * total_epoch / total_batch_size / 100
train.max_iter = int(total_epoch * 118000 / total_batch_size)
lr_multiplier = L(WarmupParamScheduler)(
    scheduler=L(MultiStepParamScheduler)(
        values=[1.0, 0.1, 0.01],
        milestones=[int(train.max_iter*24/27), int(train.max_iter*26/27)],
        num_updates=train.max_iter,
    ),
    warmup_length=250 / train.max_iter,
    warmup_factor=0.001,
)

# Optimizer
optimizer = model_zoo.get_config("common/optim.py").AdamW
optimizer.params.overrides = {"pos_embed": {"weight_decay": 0.0}}
optimizer.lr = optimizer.lr * total_batch_size / 64
optimizer.params.lr_factor_func = partial(get_vit_lr_decay_rate, lr_decay_rate=1.0, num_layers=12)
optimizer.weight_decay = 0.05
