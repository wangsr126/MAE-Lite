# MAE-lite
A Closer Look at Self-supervised Lightweight Vision Transformers [[paper]](https://arxiv.org/abs/2205.14443)

## Install
```bash
pip3 install -r requirements.txt
python3 setup.py build develop --user
```

## Preparation
Prepare the ImageNet data in `${BASE_FOLDER}/data/imagenet/imagenet_train`, `${BASE_FOLDER}/data/imagenet/imagenet_val`. Since we have an internal platform(storage) to read imagenet, I have not tried the local mode. You may need to do some modification in mae_lite/data/dataset/imagenet.py to support the local mode.

## Pretraining
```bash
# 4096 batch-sizes on 8x2080 GPUs:
cd playground/mae_lite
ssl_train -b 4096 -f mae_lite_exp.py --amp
```
## Fine-tuning on ImageNet
Please download the pre-trained models to corresponding folders, *e.g.*, 

download [MAE-lite](https://drive.google.com/file/d/1Fc8mui-dgR35hNOynWTo1gyRWw76DmPe/view?usp=sharing) to `{BASE_FOLDER}/outputs/mae_lite/mae_lite_exp/last_epoch_ckpt.pth.tar`

### Default setting:

```bash
# 1024 batch-sizes on 8x2080 GPUs:
cd playground/mae_lite
ssl_train -b 1024 -f finetuning_exp.py --amp --exp-options \
pretrain_exp_name=mae_lite/mae_lite_exp
```
### Fine-tuning with distillation (DeiT):

```bash
# 1024 batch-sizes on 8xV100 GPUs:
cd playground/mae_lite
ssl_train -b 1024 -f finetuning_distill_exp.py --amp --exp-options \
pretrain_exp_name=mae_lite/mae_lite_exp
```
### Fine-tuning with the improved recipe:

```bash
# 1024 batch-sizes on 8x2080 GPUs:
cd playground/mae_lite
ssl_train -b 1024 -f finetuning_impr_exp.py --amp --exp-options \
pretrain_exp_name=mae_lite/mae_lite_exp
```

## Transfer to Other Datasets
Please refer to [TRANSFER.md](playground/mae_lite/TRANSFER.md)

## Pre-training Distillation
Please refer to [DISTILL.md](playground/mae_lite/distill/DISTILL.md)

## Experiments of MoCo-v3
Please refer to [MOCOV3.md](playground/mocov3/MOCOV3.md)

## Main Results
|pre-train code |pre-train</br> epochs| pre-train time | fine-tune recipe | fine-tune epoch | accuracy | weights |
|---|---|---|---|---|---|---|
| - | - | - | [Default](finetuning_exp.py) | 300 | 74.5 | [ckpt](https://drive.google.com/file/d/1LADxJTuwTUBUXYGUQC9wCKJTRK4UtSl3/view?usp=sharing) |
|  |  |  | [Distillation](finetuning_distill_exp.py) | 300/1000 | 75.9/77.8 | [ckpt](https://drive.google.com/file/d/1VTnKD8y_iMaN5CQwv-MWv90AWfOP-fGp/view?usp=sharing)/[ckpt](https://drive.google.com/file/d/1LejpOPaNFziUJQYzVYroTuhlrCzXasQG/view?usp=sharing) |
|  |  |  | [Improved](finetuning_impr_exp.py) | 300 | 76.1 | [ckpt](https://drive.google.com/file/d/1QLd78alsaXHrilsvNFatbF0S8kDjTCWP/view?usp=sharing) |
| [mae_lite](mae_lite_exp.py) | 400 | ~40h | - | - | - | [ckpt](https://drive.google.com/file/d/1Fc8mui-dgR35hNOynWTo1gyRWw76DmPe/view?usp=sharing) |
|  |  |  | [Default](finetuning_exp.py) | 300 | 76.1 | [ckpt](https://drive.google.com/file/d/1jV9EaTbIxHqWNEnEWiqQG6vjT_VpT_py/view?usp=sharing) |
|  |  |  | [Distillation](finetuning_distill_exp.py) | 300/1000 | 76.9/78.4 | [ckpt](https://drive.google.com/file/d/13Wyzv7XYqxBG4az6307rXccRiP5Cbh2P/view?usp=sharing)/[ckpt](https://drive.google.com/file/d/1PoGl4QYnVpZjFnexgG4ZJkzYksDsS8Tj/view?usp=sharing) |
|  |  |  | [Improved](finetuning_impr_exp.py) | 300/1000 | 78.1/78.5 | [ckpt](https://drive.google.com/file/d/1KzX1BA1ZHhXXCDUPuA9TCqN9ic8j6IKj/view?usp=sharing)/[ckpt](https://drive.google.com/file/d/1AnqEH0qa9AnbvU46gDhk6R0OIcGDwZ7d/view?usp=sharing) |
| [mae_lite_distill](distill/mae_lite_distill_exp.py) | 400 | ~44h | - | - | - | [ckpt](https://drive.google.com/file/d/1M2FEe3SjnhcIodcoeB9uJ0v0nV5Hrlxg/view?usp=sharing) |
|  |  |  | [Default](finetuning_exp.py) | 300 | 76.5 | [ckpt](https://drive.google.com/file/d/1mSyNcaaumEm07nD_VHvkOq34Ypvj4YJL/view?usp=sharing) |

<!-- ## Citation
Please cite the following paper if this repo helps your research:
```bibtex
``` -->

## Acknowledge
We thank for the code implementation from [timm](https://github.com/rwightman/pytorch-image-models), [MAE](https://github.com/facebookresearch/mae/tree/main), [MoCo-v3](https://github.com/facebookresearch/moco-v3).


## License
This repo is released under the Apache 2.0 license. Please see the LICENSE file for more information.
