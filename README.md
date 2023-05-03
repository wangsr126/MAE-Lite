# MAE-lite
A Closer Look at Self-supervised Lightweight Vision Transformers [[paper]]()

## Install
```bash
pip3 install -r requirements.txt
python3 setup.py build develop --user
```

## Preparation
Prepare the ImageNet data in `${BASE_FOLDER}/data/imagenet/imagenet_train`, `${BASE_FOLDER}/data/imagenet/imagenet_val`.

## Pretraining
```bash
# 4096 batch-sizes on 8x2080 GPUs:
cd projects/mae_lite
ssl_train -b 4096 -d 0-7 -e 400 -f mae_lite_exp.py --amp \
--exp-options exp_name=mae_lite/mae_tiny_400e
```
## Fine-tuning on ImageNet
Please download the pre-trained models, *e.g.*, 

download [MAE-Tiny](https://drive.google.com/file/d/1ZQYlvCPLZrJDqn2lp4GCIVL246WPqgEf/view?usp=sharing) to `{BASE_FOLDER}/checkpoints/mae_tiny_400e.pth.tar`

### Fine-tuning with the improved recipe:

```bash
# 1024 batch-sizes on 8x2080 GPUs:
cd projects/eval_tools
ssl_train -b 1024 -d 0-7 -e 300 -f finetuning_exp.py --amp \
--ckpt $CKPT --exp-options pretrain_exp_name=mae_lite/mae_tiny_400e
```
- `CKPT`: if set to "{BASE_FOLDER}/checkpoints/mae_tiny_400e.pth.tar", it will be loaded as initialization; If not set, the checkpoint at "{BASE_FOLDER}/outputs/mae_lite/mae_tiny_400e/last_epoch_ckpt.pth.tar" will be loaded automatically.

## Pre-training with Distillation
Please refer to [DISTILL.md](projects/mae_lite/DISTILL.md)

## Transfer to Other Datasets
Please refer to [TRANSFER.md](projects/eval_tools/TRANSFER.md)

## Transfer to Detection Tasks
Please refer to [DETECTION.md](projects/eval_tools/det/DETECTION.md)

## Experiments of MoCo-v3
Please refer to [MOCOV3.md](projects/mocov3/MOCOV3.md)

## Main Results
|pre-train code |pre-train</br> epochs| fine-tune recipe | fine-tune epoch | accuracy | ckpt |
|---|---|---|---|---|---|
| - | - | [impr.](projects/eval_tools/finetuning_exp.py) | 300 | 75.8 | [link](https://drive.google.com/file/d/1RvhE2HucdWYHhKmPfHQW2A4EPpCHSYN_/view?usp=sharing) |
| [mae_lite](projects/mae_lite/mae_lite_exp.py) | 400 | - | - | - | [link](https://drive.google.com/file/d/1ZQYlvCPLZrJDqn2lp4GCIVL246WPqgEf/view?usp=sharing) |
|  |  | [impr.](projects/eval_tools/finetuning_exp.py) | 300 | 78.1 | [link](https://drive.google.com/file/d/1VEpG2c5A62PefeecjQ3yaKfRlfph3LxO/view?usp=sharing) |
|  |  | [impr.+RPE](projects/eval_tools/finetuning_rpe_exp.py) | 1000 | 79.0 | [link](https://drive.google.com/file/d/1zKDnMKs6tBTnC4liTYG2AMtotKcbKr4J/view?usp=sharing) |
| [mae_lite_distill](projects/mae_lite/mae_lite_distill_exp.py) | 400 | - | - | - | [link](https://drive.google.com/file/d/1OCDMUEdcPhwoCPWGN0kahsHST7tbQmFe/view?usp=sharing) |
|  |  | [impr.](projects/eval_tools/finetuning_exp.py) | 300 | 78.4 | [link](https://drive.google.com/file/d/1bcxwRUx6fq38M9eoBQbP2thwtU0j_9u6/view?usp=sharing) |

<!-- ## Citation
Please cite the following paper if this repo helps your research:
```bibtex
``` -->

## Acknowledge
We thank for the code implementation from [timm](https://github.com/rwightman/pytorch-image-models), [MAE](https://github.com/facebookresearch/mae/tree/main), [MoCo-v3](https://github.com/facebookresearch/moco-v3).


## License
This repo is released under the Apache 2.0 license. Please see the LICENSE file for more information.
