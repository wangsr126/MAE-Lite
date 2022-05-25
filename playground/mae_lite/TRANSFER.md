# Transfer to Other Datasets
This folder includes the transfer learning experiments on CIFAR-100, Flowers, Pets, Aircraft, Inatualist18 and Cars datasets.

## Preparation
Please download pre-trained models to corresponding folders, *e.g.*: 

Download [MAE-lite](https://drive.google.com/file/d/1Fc8mui-dgR35hNOynWTo1gyRWw76DmPe/view?usp=sharing) to `{BASE_FOLDER}/outputs/mae_lite/mae_lite_exp/last_epoch_ckpt.pth.tar`

Download [DeiT](https://drive.google.com/file/d/1LADxJTuwTUBUXYGUQC9wCKJTRK4UtSl3/view?usp=sharing)-`{BASE_FOLDER}/outputs/mae_lite/deit_exp/last_epoch_ckpt.pth.tar`

The datasets will be downloaded automatically.

## Scripts
512 batch-sizes on 4 $\times$ 2080 GPUs:
```bash
cd playground/mae_lite
ssl_train -b 512 -d 0-3 -e $EPOCH$ -f finetuning_transfer_exp.py --amp --exp-options \
pretrain_exp_name=$EXPN$ dataset=$DATASET$ warmup_epochs=$WARMUP_EPOCHS$ save_folder_prefix=$SAVE_FOLDER_PREFIX$ lr=$LR$ layer_decay=$LAYER_DECAY$
```
- `EXPN`: `mae_lite/mae_lite_exp`, `mae_lite/deit_exp`
- `DATASET`: `CIFAR10`, `CIFAR100`, `Aircraft`, `INatDataset`, `Flowers`, `Pets`, `Cars`
- `WARMUP_EPOCHS`: warmup epochs
- `SAVE_FOLDER_PREFIX`: we will save logs and ckpts to `$BASE_FOLDER$/outputs/$EXPN$/$SAVE_FOLDER_PREFIX$eval/`
- `LR`: learning rate
- `LAYER_DECAY`: layer-wise lr decay