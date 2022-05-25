# Pre-training Distillation

## Preparation
Download the teacher [MAE-Base](https://drive.google.com/file/d/1Jf-UE2pRx-qe4r__gzx-WAYZsaFu-BOj/view?usp=sharing) to `{BASE_FOLDER}/checkpoints/mae_base_1600e.pth.tar`
## Pre-training
```bash
# 4096 batch-sizes on 8xV100 GPUs:
cd playground/mae_lite
ssl_train -b 4096 -f distill/mae_lite_distill_exp.py --amp --exp-options \
teacher_ckpt_path=$CKPT_PATH
```
- `$CKPT_PATH`: `{BASE_FOLDER}/checkpoints/mae_base_1600e.pth.tar`
## Fine-tuning
### Default setting:
```bash
# 1024 batch-sizes on 8x2080 GPUs:
cd playground/mae_lite
ssl_train -b 1024 -f finetuning_exp.py --amp --exp-options \
pretrain_exp_name=mae_lite/distill/mae_lite_distill_exp
```