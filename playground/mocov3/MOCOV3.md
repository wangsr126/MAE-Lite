# Experiments on MoCo-v3

## Scripts
### Pretraining
```bash
# 1024 batch-sizes on 8xV100 GPUs:
cd playground/mocov3
ssl_train -b 1024 -f ./mocov3_exp.py --amp
```
### Finetuning
```bash
# 1024 batch-sizes on 8x2080 GPUs:
cd playground/mae_lite
ssl_train -b 1024 -f finetuning_exp.py --amp --exp-options \
pretrain_exp_name=mocov3/mocov3_exp layer_decay=0.75 global_pool=False weights_prefix=base_encoder
```