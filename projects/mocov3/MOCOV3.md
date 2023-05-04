## Experiments on [MoCo-v3](https://arxiv.org/abs/2104.02057)

### Pre-Training
```bash
# 1024 batch-sizes on 8 GPUs:
cd projects/mocov3
ssl_train -b 1024 -d 0-7 -e 400 -f ./mocov3_exp.py --amp \
--exp-options exp_name=mocov3/mocov3_tiny_400e
```

### Fine-Tuning

```bash
# 1024 batch-sizes on 8 GPUs:
cd projects/eval_tools
ssl_train -b 1024 -d 0-7 -e 300 -f finetuning_exp.py --amp \
[--ckpt <checkpoint-path>] \
--exp-options pretrain_exp_name=mocov3/mocov3_tiny_400e layer_decay=0.75 weights_prefix=base_encoder
```

### Main Results
|pre-train code |pre-train</br> epochs| fine-tune recipe | fine-tune epoch | accuracy | ckpt |
|---|---|---|---|---|---|
| [mocov3](mocov3_exp.py) | 400 | - | - | - | [link](https://drive.google.com/file/d/1RI0mU-PweAVIXs_hNOx-Xw3VRhN7w6un/view?usp=sharing) |
|  |  | [impr.](../eval_tools/finetuning_exp.py) | 300 | 76.8 | [link](https://drive.google.com/file/d/1WxEQxFhnt6vMZ08ZArof41fAIKRefziE/view?usp=sharing) |