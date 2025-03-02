## Transfer to ADE20k Semantic Segmentation Tasks
Transfer evaluation of the pre-trained models on ADE20k Semantic Segmentation tasks based on [Semantic-FPN](https://arxiv.org/abs/1901.02446) and [MMSegmentation](https://github.com/open-mmlab/mmsegmentation).

### Preparation
Install [mmcv-full](https://github.com/open-mmlab/mmcv) and [MMSegmentation v1.1.1](https://github.com/open-mmlab/mmsegmentation/tree/v1.1.1). Later versions should work as well. The easiest way is to install via [MIM](https://github.com/open-mmlab/mim).
```
pip install -U openmim
mim install mmengine
mim install "mmcv==2.0.1"
pip install "mmsegmentation==1.1.1"
```

Please following the [insructions in MMSeg](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#prepare-datasets) to prepare the datasets.

Then download the pre-trained models, *e.g.*: 

* Download [DeiT-Tiny](https://drive.google.com/file/d/1RvhE2HucdWYHhKmPfHQW2A4EPpCHSYN_/view?usp=sharing) to `<BASE_FOLDER>/checkpoints/deit_tiny_300e.pth.tar`

* Download [MAE-Distill-D²-Tiny](https://drive.google.com/file/d/1gOKB8lSQ3IlOLNbi5Uc7saZO2mMClXNU/view?usp=sharing) to `<BASE_FOLDER>/checkpoints/mae_tiny_distill_d2_400e.pth.tar`

* Download [DeiT-Tiny-FT-RPE](https://drive.google.com/file/d/1DzRgVyXRx9m0P1-AxA5GPyafgPTipYke/view?usp=sharing) to `<BASE_FOLDER>/checkpoints/deit_tiny_rpe_1000e.pth.tar`

* Download [MAE-Tiny-Distill-D²-FT-RPE](https://drive.google.com/file/d/1KRdkurYMfNxaIhjn3bELLL40Z-MEbMZj/view?usp=sharing) to `<BASE_FOLDER>/checkpoints/mae_tiny_distill_d2_400e_ft_rpe_1000e.pth.tar`

And convert them:
```bash
cd projects/eval_tools/seg
python tools/model_converters/beit2mmseg.py <BASE_FOLDER>/checkpoints/deit_tiny_300e.pth.tar pretrain/deit_tiny_300e_mmcls.pth
python tools/model_converters/beit2mmseg.py <BASE_FOLDER>/checkpoints/mae_tiny_distill_d2_400e.pth.tar pretrain/mae_tiny_distill_d2_400e_mmcls.pth
python tools/model_converters/beit2mmseg.py <BASE_FOLDER>/checkpoints/deit_tiny_rpe_1000e.pth.tar pretrain/deit_tiny_rpe_1000e_mmcls.pth 
python tools/model_converters/beit2mmseg.py <BASE_FOLDER>/checkpoints/mae_tiny_distill_d2_400e_ft_rpe_1000e.pth.tar pretrain/mae_tiny_distill_d2_400e_ft_rpe_1000e_mmcls.pth 
```

### Fine-Tuning on ADE20k
```bash
# on 8 GPUs:
cd projects/eval_tools/seg
bash tools/dist_train.sh ./configs/sem_fpn/semantic_fpn_t_160k.py 8
```

## Main Results
By default, the configs use `image_size=512`.
|Config|Initialization | mIoU | ckpt | 
|---|---|---|---|
|[semantic_fpn_t_160k.py](configs/sem_fpn/semantic_fpn_t_160k.py)| [DeiT-Tiny](https://drive.google.com/file/d/1RvhE2HucdWYHhKmPfHQW2A4EPpCHSYN_/view?usp=sharing) | 38.0 | [link](https://drive.google.com/file/d/1f2670Le8rwH2usrkEfHqgb4K4FfC89o_/view?usp=sharing) |
|[semantic_fpn_t_160k_mae_d2_distill.py](configs/sem_fpn/semantic_fpn_t_160k_mae_d2_distill.py)| [MAE-Distill-D²-Tiny](https://drive.google.com/file/d/1gOKB8lSQ3IlOLNbi5Uc7saZO2mMClXNU/view?usp=sharing) | 39.0 | [link](https://drive.google.com/file/d/1Y6GA6oGlpTYATJysvjqsAADLj11oGFsk/view?usp=drive_link) |
|[semantic_fpn_t_160k.py](configs/sem_fpn/semantic_fpn_t_160k.py)| [DeiT-Tiny-FT-RPE](https://drive.google.com/file/d/1DzRgVyXRx9m0P1-AxA5GPyafgPTipYke/view?usp=sharing) | 41.5 | [link](https://drive.google.com/file/d/1FkFpCqBTODfQyj2d9_fx77p4XX7b5AaA/view?usp=drive_link) |
|[semantic_fpn_t_160k_mae_d2_distill.py](configs/sem_fpn/semantic_fpn_t_160k_mae_d2_distill.py)| [MAE-Tiny-Distill-D²-FT-RPE](https://drive.google.com/file/d/1KRdkurYMfNxaIhjn3bELLL40Z-MEbMZj/view?usp=sharing) | 42.8 | [link](https://drive.google.com/file/d/10HQX0CAS1BsgEwZkTKeIzntfbunCyxXn/view?usp=drive_link) |

