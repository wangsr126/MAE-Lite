# Transfer to Other Datasets
This folder includes the transfer learning experiments on CIFAR-100, Flowers, Pets, Aircraft and Cars datasets.

## Preparation
Please download the pre-trained models, *e.g.*: 

Download [DeiT-Tiny](https://drive.google.com/file/d/1RvhE2HucdWYHhKmPfHQW2A4EPpCHSYN_/view?usp=sharing) to `{BASE_FOLDER}/checkpoints/deit_tiny_300e.pth.tar`

Download [MAE-Tiny](https://drive.google.com/file/d/1ZQYlvCPLZrJDqn2lp4GCIVL246WPqgEf/view?usp=sharing) to `{BASE_FOLDER}/checkpoints/mae_tiny_400e.pth.tar`

Download [MoCov3-Tiny](https://drive.google.com/file/d/1RI0mU-PweAVIXs_hNOx-Xw3VRhN7w6un/view?usp=sharing) to `{BASE_FOLDER}/checkpoints/mocov3_tiny_300e.pth.tar`

The datasets will be downloaded automatically.

## Scripts
512 batch-sizes on 4 $\times$ 2080 GPUs:
```bash
cd projects/eval_tools
ssl_train -b 512 -d 0-3 -e $EPOCH -f finetuning_transfer_exp.py --amp \
--ckpt $CKPT --exp-options pretrain_exp_name=$EXPN \
dataset=$DATASET warmup_epochs=$WARMUP_EPOCHS save_folder_prefix=$PREFIX lr=$LR layer_decay=$LAYER_DECAY
```
- `EPOCH`: number of fine-tuning epochs
- `CKPT`: path to the checkpoint used as initialization, *e.g.*, `{BASE_FOLDER}/checkpoints/deit_tiny_300e.pth.tar`
- `DATASET`: `CIFAR10`, `CIFAR100`, `Aircraft`, `Flowers`, `Pets`, `Cars`
- `WARMUP_EPOCHS`: warmup epochs
- `EXPN`/`PREFIX`: we will save logs and ckpts to `$BASE_FOLDER/outputs/$EXPN/$PREFIXeval/`
- `LR`: learning rate
- `LAYER_DECAY`: layer-wise lr decay

Please refer to [transfer.sh](scripts/transfer.sh) for detailed script examples.