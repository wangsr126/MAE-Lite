## Pre-Training with Distillation

### Preparation
Download the teacher [MAE-Base](https://drive.google.com/file/d/1SPTjHIvw-yTOmw2ll-9cCiVyqR8NdPrX/view?usp=sharing) to `<BASE_FOLDER>/checkpoints/mae_base_1600e.pth.tar`

### Pre-Training
```bash
# 4096 batch-sizes on 8 GPUs:
cd projects/mae_lite
ssl_train -b 4096 -f mae_lite_distill_exp.py --amp --exp-options \
teacher_ckpt_path="<BASE_FOLDER>/checkpoints/mae_base_1600e.pth.tar" exp_name=mae_lite/mae_tiny_distill_400e
```

```bash
# Decoupled Distillation (D²-MAE):
cd projects/mae_lite
ssl_train -b 4096 -f mae_lite_distill_d2_exp.py --amp --exp-options \
teacher_ckpt_path="<BASE_FOLDER>/checkpoints/mae_base_1600e.pth.tar" exp_name=mae_lite/mae_tiny_distill_400e
```

### Fine-Tuning
```bash
# 1024 batch-sizes on 8 GPUs:
cd projects/eval_tools
ssl_train -b 1024 -f finetuning_exp.py --amp [--ckpt <checkpoint-path>] \
--exp-options pretrain_exp_name=mae_lite/mae_tiny_distill_400e
```

# Main Results
|pre-train code |pre-train</br> epochs| fine-tune recipe | fine-tune epoch | accuracy | ckpt |
|---|---|---|---|---|---|
| [mae_lite_distill](mae_lite_distill_exp.py) | 400 | - | - | - | [link](https://drive.google.com/file/d/1OCDMUEdcPhwoCPWGN0kahsHST7tbQmFe/view?usp=sharing) |
| [mae_lite_distill](mae_lite_distill_exp.py) | 400 | [impr.](../eval_tools/finetuning_exp.py) | 300 | 78.4 | [link](https://drive.google.com/file/d/1bcxwRUx6fq38M9eoBQbP2thwtU0j_9u6/view?usp=sharing) |
| [mae_lite_d2_distill](mae_lite_distill_d2_exp.py) | 400 | - | - | - | [link](https://drive.google.com/file/d/1gOKB8lSQ3IlOLNbi5Uc7saZO2mMClXNU/view?usp=sharing) |
| [mae_lite_d2_distill](mae_lite_distill_d2_exp.py) | 400 | [impr.](../eval_tools/finetuning_exp.py) | 300 | 78.7 | [link](https://drive.google.com/file/d/1c1KCRQ1MBtW6Zw4Uq1yAcv6zuYGcoWaW/view?usp=sharing) |
