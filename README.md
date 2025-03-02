# [MAE-Lite](https://arxiv.org/abs/2404.12210) (IJCV 2025)
<p>
<a href="https://link.springer.com/article/10.1007/s11263-024-02327-w"><img src="https://img.shields.io/badge/publication-Paper-<color>"></a> 
<a href="https://arxiv.org/abs/2404.12210" alt="arXiv">
    <img src="https://img.shields.io/badge/arXiv-2404.12210-b31b1b.svg?style=flat" /></a>
</p>

<p>
  <a href="#news">News</a> |
  <a href="#introduction">Introduction</a> |
  <a href="#getting_started">Getting Started</a> |
  <a href="#main_results">Main Results</a> |
  <a href="#citation">Citation</a> |
  <a href="#acknowledge">Acknowledge</a>
</p >

> [**An Experimental Study on Exploring Strong Lightweight Vision Transformers via Masked Image Modeling Pre-training**](https://arxiv.org/abs/2404.12210)  
> Jin Gao, Shubo Lin, Shaoru Wang*, Yutong Kou, Zeming Li, Liang Li, Congxuan Zhang, Xiaoqin Zhang, Yizheng Wang, Weiming Hu   
> *IJCV 2025*

> [**A Closer Look at Self-Supervised Lightweight Vision Transformers**](https://arxiv.org/abs/2205.14443)  
> Shaoru Wang, Jin Gao*, Zeming Li, Xiaoqin Zhang, Weiming Hu  
> *ICML 2023*

<h2 id="news">🎉 News</h2> 

* **`2024.12`:** Our extended version is accepted by *IJCV 2025*!
* **`2023.5`:** Code & models are released!
* **`2023.4`:** Our paper is accepted by *ICML 2023*!
* **`2022.5`:** Our initial version of the paper was published on Arxiv.

<h2 id="introduction">✨ Introduction</h2> 

**MAE-Lite** focuses on exploring the pre-training of lightweight Vision Transformers (ViTs). This repo provide the code and models for the study in the paper.
* We provide advanced pre-training (based on [MAE](https://arxiv.org/abs/2111.06377)) and fine-tuning recipes for lightweight ViTs and demonstrate that *even vanilla lightweight ViT (*e.g.*, ViT-Tiny) beats most previous SOTA ConvNets and ViT derivatives with delicate network architecture design*. We achieve **79.0%** top-1 accuracy on ImageNet with vanilla ViT-Tiny (5.7M).
* We provide code for the transfer evaluation of pre-trained models on several classification tasks (*e.g.*, [Oxford 102 Flower](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/), [Oxford-IIIT Pet](https://www.robots.ox.ac.uk/~vgg/data/pets/), [FGVC Aircraft](https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/), [CIFAR](https://www.cs.toronto.edu/~kriz/cifar.html), *etc.*) and COCO detection tasks (based on [ViTDet](https://github.com/facebookresearch/detectron2/blob/main/projects/ViTDet)). We find that *the self-supervised pre-trained ViTs work worse than the supervised pre-trained ones on data-insufficient downstream tasks*.
* We provide code for the analysis tools used in the paper to examine the layer representations and attention distance & entropy for the ViTs.
* We provide code and models for our proposed knowledge distillation method for the pre-trained lightweight ViTs based on MAE, which shows superiority on the trasfer evaluation of data-insufficient classification tasks and dense prediction tasks.

***update (2025.02.28)*** 
* We provide benchmark for more masked image modeling (MIM) pre-training methods ([BEiT](https://github.com/microsoft/unilm/tree/master/beit), [BootMAE](https://github.com/LightDXY/BootMAE.git), [MaskFeat](https://github.com/facebookresearch/SlowFast/blob/main/projects/maskfeat/README.md)) on lightweight ViTs and evaluate their transferability to [downstream tasks](projects/eval_tools/TRANSFER.md).
* We provide code and models for our [decoupled distillation method](projects/mae_lite/DISTILL.md) during pre-training and transfer to more dense prediction tasks including [detection](projects/eval_tools/det/DETECTION.md), [tracking](projects/eval_tools/track/TRACKING.md) and [semantic segmentation](projects/eval_tools/seg/SEGMENTATION.md), which enables SOTA performance on the ADE20K segmentation task (**42.8%** mIoU) and LaSOT tracking task (**66.1%** AUC) in the lightweight regime. The latter even surpasses all the current SOTA lightweight CPU-realtime trackers.
* We extend our distillation method to hierarchical ViTs ([Swin](projects/simmim/SIMMIM.md) and [Hiera](projects/mae_lite/HIERA.md)), which validate the generalizability and effectiveness of the distillation following our observation-analysis-solution flow.

<h2 id="getting_started">📋 Getting Started</h2> 

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

### Evaluation of fine-tuned models
download [MAE-Tiny-FT](https://drive.google.com/file/d/1VEpG2c5A62PefeecjQ3yaKfRlfph3LxO/view?usp=sharing) to `<BASE_FOLDER>/checkpoints/mae_tiny_400e_ft_300e.pth.tar`

```bash
# 1024 batch-sizes on 1 GPUs:
python mae_lite/tools/eval.py -b 1024 -d 0 -f projects/eval_tools/finetuning_exp.py \
--ckpt <BASE_FOLDER>/checkpoints/mae_tiny_400e_ft_300e.pth.tar \
--exp-options pretrain_exp_name=mae_lite/mae_tiny_400e/ft_eval
```

And you will get `"Top1: 77.978"` if all right.

download [MAE-Tiny-FT-RPE](https://drive.google.com/file/d/1zKDnMKs6tBTnC4liTYG2AMtotKcbKr4J/view?usp=sharing) to `<BASE_FOLDER>/checkpoints/mae_tiny_400e_ft_rpe_1000e.pth.tar`

```bash
# 1024 batch-sizes on 1 GPUs:
python mae_lite/tools/eval.py -b 1024 -d 0 -f projects/eval_tools/finetuning_rpe_exp.py \
--ckpt <BASE_FOLDER>/checkpoints/mae_tiny_400e_ft_rpe_1000e.pth.tar \
--exp-options pretrain_exp_name=mae_lite/mae_tiny_400e/ft_rpe_eval
```

And you will get `"Top1: 79.002"` if all right.

download [MAE-Tiny-Distill-D²-FT-RPE](https://drive.google.com/file/d/1KRdkurYMfNxaIhjn3bELLL40Z-MEbMZj/view?usp=sharing) to `<BASE_FOLDER>/checkpoints/mae_tiny_distill_d2_400e_ft_rpe_1000e.pth.tar`

```bash
# 1024 batch-sizes on 1 GPUs:
python mae_lite/tools/eval.py -b 1024 -d 0 -f projects/eval_tools/finetuning_rpe_exp.py \
--ckpt <BASE_FOLDER>/checkpoints/mae_tiny_distill_d2_400e_ft_rpe_1000e.pth.tar \
--exp-options pretrain_exp_name=mae_lite/mae_tiny_400e/ft_rpe_eval qv_bias=False
```

And you will get `"Top1: 79.444"` if all right.

### Pre-Training with Distillation
Please refer to [DISTILL.md](projects/mae_lite/DISTILL.md).

### Transfer to Other Datasets
Please refer to [TRANSFER.md](projects/eval_tools/TRANSFER.md).

### Transfer to Detection Tasks
Please refer to [DETECTION.md](projects/eval_tools/det/DETECTION.md).

### Transfer to Tracking Tasks
Please refer to [TRACKING.md](projects/eval_tools/track/TRACKING.md).

### Transfer to Semantic Segmentation Tasks
Please refer to [SEGMENTATION.md](projects/eval_tools/seg/SEGMENTATION.md).

### Experiments of MoCo-v3
Please refer to [MOCOV3.md](projects/mocov3/MOCOV3.md).

### Models Analysis Tools
Please refer to [VISUAL.md](projects/eval_tools/VISUAL.md).

<h2 id="main_results">📄 Main Results</h2> 

|pre-train code |pre-train</br> epochs| fine-tune recipe | fine-tune epoch | accuracy | ckpt |
|---|---|---|---|---|---|
| - | - | [impr.](projects/eval_tools/finetuning_exp.py) | 300 | 75.8 | [link](https://drive.google.com/file/d/1RvhE2HucdWYHhKmPfHQW2A4EPpCHSYN_/view?usp=sharing) |
| [mae_lite](projects/mae_lite/mae_lite_exp.py) | 400 | - | - | - | [link](https://drive.google.com/file/d/1ZQYlvCPLZrJDqn2lp4GCIVL246WPqgEf/view?usp=sharing) |
|  |  | [impr.](projects/eval_tools/finetuning_exp.py) | 300 | 78.0 | [link](https://drive.google.com/file/d/1VEpG2c5A62PefeecjQ3yaKfRlfph3LxO/view?usp=sharing) |
|  |  | [impr.+RPE](projects/eval_tools/finetuning_rpe_exp.py) | 1000 | **79.0** | [link](https://drive.google.com/file/d/1zKDnMKs6tBTnC4liTYG2AMtotKcbKr4J/view?usp=sharing) |
| [mae_lite_distill](projects/mae_lite/mae_lite_distill_exp.py) | 400 | - | - | - | [link](https://drive.google.com/file/d/1OCDMUEdcPhwoCPWGN0kahsHST7tbQmFe/view?usp=sharing) |
|  |  | [impr.](projects/eval_tools/finetuning_exp.py) | 300 | 78.4 | [link](https://drive.google.com/file/d/1bcxwRUx6fq38M9eoBQbP2thwtU0j_9u6/view?usp=sharing) |
| [mae_lite_d2_distill](projects/mae_lite/mae_lite_distill_d2_exp.py) | 400 | - | - | - | [link](https://drive.google.com/file/d/1gOKB8lSQ3IlOLNbi5Uc7saZO2mMClXNU/view?usp=sharing) |
|  |  | [impr.](projects/eval_tools/finetuning_exp.py) | 300 | 78.7 | [link](https://drive.google.com/file/d/1c1KCRQ1MBtW6Zw4Uq1yAcv6zuYGcoWaW/view?usp=sharing) |
|  |  | [impr.+RPE](projects/eval_tools/finetuning_rpe_exp.py) | 1000 | **79.4** | [link](https://drive.google.com/file/d/1KRdkurYMfNxaIhjn3bELLL40Z-MEbMZj/view?usp=sharing) |

<h2 id="citation">🏷️ Citation</h2> 

Please cite the following paper if this repo helps your research:
```bibtex
@misc{wang2023closer,
      title={A Closer Look at Self-Supervised Lightweight Vision Transformers}, 
      author={Shaoru Wang and Jin Gao and Zeming Li and Xiaoqin Zhang and Weiming Hu},
      journal={arXiv preprint arXiv:2205.14443},
      year={2023},
}

@article{gao2025experimental,
      title={An Experimental Study on Exploring Strong Lightweight Vision Transformers via Masked Image Modeling Pre-training},
      author={Jin Gao, Shubo Lin, Shaoru Wang, Yutong Kou, Zeming Li, Liang Li, Congxuan Zhang, Xiaoqin Zhang, Yizheng Wang, Weiming Hu},
      journal={International Journal of Computer Vision},
      year={2025},
      doi={10.1007/s11263-024-02327-w},
      publisher={Springer}
}
```

<h2 id="acknowledge">🤝 Acknowledge</h2> 

We thank for the code implementation from [timm](https://github.com/rwightman/pytorch-image-models), [MAE](https://github.com/facebookresearch/mae/tree/main), [MoCo-v3](https://github.com/facebookresearch/moco-v3).


## License
This repo is released under the Apache 2.0 license. Please see the LICENSE file for more information.
