# MAE-Lite
> [**A Closer Look at Self-Supervised Lightweight Vision Transformers**](https://arxiv.org/abs/2205.14443)  
> Shaoru Wang, Jin Gao*, Zeming Li, Xiaoqin Zhang, Weiming Hu  
> *ICML 2023*

## News
* **`2023.5`:** Code & models are released!
* **`2023.4`:** Our paper is accepted by *ICML 2023*!
* **`2022.5`:** Our initial version of the paper was published on Arxiv.

## Introduction
**MAE-Lite** focuses on exploring the pre-training of lightweight Vision Transformers (ViTs). This repo provide the code and models for the study in the paper.
* We provide advanced pre-training (based on [MAE](https://arxiv.org/abs/2111.06377)) and fine-tuning recipes for lightweight ViTs and demonstrate that *even vanilla lightweight ViT (*e.g.*, ViT-Tiny) beats most previous SOTA ConvNets and ViT derivatives with delicate network architecture design*. We achieve **79.0%** top-1 accuracy on ImageNet with vanilla ViT-Tiny (5.7M).
* We provide code for the transfer evaluation of pre-trained models on several classification tasks (*e.g.*, [Oxford 102 Flower](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/), [Oxford-IIIT Pet](https://www.robots.ox.ac.uk/~vgg/data/pets/), [FGVC Aircraft](https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/), [CIFAR](https://www.cs.toronto.edu/~kriz/cifar.html), *etc.*) and COCO detection tasks (based on [ViTDet](https://github.com/facebookresearch/detectron2/blob/main/projects/ViTDet)). We find that *the self-supervised pre-trained ViTs work worse than the supervised pre-trained ones on data-insufficient downstream tasks*.
* We provide code for the analysis tools used in the paper to examine the layer representations and attention distance & entropy for the ViTs.
* We provide code and models for our proposed knowledge distillation method for the pre-trained lightweight ViTs based on MAE, which shows superiority on the trasfer evaluation of data-insufficient classification tasks and dense prediction tasks.

## Getting Started

### Installation
Setup conda environment:
```bash
# Create environment
conda create -n mae-lite python=3.7 -y
conda activate mae-lite

# Instaill requirements
conda install pytorch==1.9.0 torchvision==0.10.0 -c pytorch -y

# Clone MAE-Lite
git clone https://github.com/wangsr126/mae-lite.git
cd mae-lite

# Install other requirements
pip3 install -r requirements.txt
python3 setup.py build develop --user
```

### Data Preparation
Prepare the ImageNet data in `<BASE_FOLDER>/data/imagenet/imagenet_train`, `<BASE_FOLDER>/data/imagenet/imagenet_val`.

### Pre-Training
To pre-train ViT-Tiny with our recommended MAE recipe:
```bash
# 4096 batch-sizes on 8 GPUs:
cd projects/mae_lite
ssl_train -b 4096 -d 0-7 -e 400 -f mae_lite_exp.py --amp \
--exp-options exp_name=mae_lite/mae_tiny_400e
```

### Fine-Tuning on ImageNet
Please download the pre-trained models, *e.g.*, 

download [MAE-Tiny](https://drive.google.com/file/d/1ZQYlvCPLZrJDqn2lp4GCIVL246WPqgEf/view?usp=sharing) to `<BASE_FOLDER>/checkpoints/mae_tiny_400e.pth.tar`

To fine-tune with the improved recipe:

```bash
# 1024 batch-sizes on 8 GPUs:
cd projects/eval_tools
ssl_train -b 1024 -d 0-7 -e 300 -f finetuning_exp.py --amp \
[--ckpt <checkpoint-path>] --exp-options pretrain_exp_name=mae_lite/mae_tiny_400e
```
- `<checkpoint-path>`: if set to `<BASE_FOLDER>/checkpoints/mae_tiny_400e.pth.tar`, it will be loaded as initialization; If not set, the checkpoint at `<BASE_FOLDER>/outputs/mae_lite/mae_tiny_400e/last_epoch_ckpt.pth.tar` will be loaded automatically.

### Pre-training with Distillation
Please refer to [DISTILL.md](projects/mae_lite/DISTILL.md).

### Transfer to Other Datasets
Please refer to [TRANSFER.md](projects/eval_tools/TRANSFER.md).

### Transfer to Detection Tasks
Please refer to [DETECTION.md](projects/eval_tools/det/DETECTION.md).

### Experiments of MoCo-v3
Please refer to [MOCOV3.md](projects/mocov3/MOCOV3.md).

### Models Analysis Tools
Please refer to [VISUAL.md](projects/eval_tools/VISUAL.md).

## Main Results
|pre-train code |pre-train</br> epochs| fine-tune recipe | fine-tune epoch | accuracy | ckpt |
|---|---|---|---|---|---|
| - | - | [impr.](projects/eval_tools/finetuning_exp.py) | 300 | 75.8 | [link](https://drive.google.com/file/d/1RvhE2HucdWYHhKmPfHQW2A4EPpCHSYN_/view?usp=sharing) |
| [mae_lite](projects/mae_lite/mae_lite_exp.py) | 400 | - | - | - | [link](https://drive.google.com/file/d/1ZQYlvCPLZrJDqn2lp4GCIVL246WPqgEf/view?usp=sharing) |
|  |  | [impr.](projects/eval_tools/finetuning_exp.py) | 300 | 78.1 | [link](https://drive.google.com/file/d/1VEpG2c5A62PefeecjQ3yaKfRlfph3LxO/view?usp=sharing) |
|  |  | [impr.+RPE](projects/eval_tools/finetuning_rpe_exp.py) | 1000 | **79.0** | [link](https://drive.google.com/file/d/1zKDnMKs6tBTnC4liTYG2AMtotKcbKr4J/view?usp=sharing) |
| [mae_lite_distill](projects/mae_lite/mae_lite_distill_exp.py) | 400 | - | - | - | [link](https://drive.google.com/file/d/1OCDMUEdcPhwoCPWGN0kahsHST7tbQmFe/view?usp=sharing) |
|  |  | [impr.](projects/eval_tools/finetuning_exp.py) | 300 | 78.4 | [link](https://drive.google.com/file/d/1bcxwRUx6fq38M9eoBQbP2thwtU0j_9u6/view?usp=sharing) |

## Citation
Please cite the following paper if this repo helps your research:
```bibtex
@misc{wang2023closer,
      title={A Closer Look at Self-Supervised Lightweight Vision Transformers}, 
      author={Shaoru Wang and Jin Gao and Zeming Li and Xiaoqin Zhang and Weiming Hu},
      journal={arXiv preprint arXiv:2205.14443},
      year={2023},
}
```

## Acknowledge
We thank for the code implementation from [timm](https://github.com/rwightman/pytorch-image-models), [MAE](https://github.com/facebookresearch/mae/tree/main), [MoCo-v3](https://github.com/facebookresearch/moco-v3).


## License
This repo is released under the Apache 2.0 license. Please see the LICENSE file for more information.
