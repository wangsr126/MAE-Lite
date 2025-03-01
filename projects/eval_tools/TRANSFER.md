## Transfer to Other Datasets
This folder includes the transfer learning experiments on CIFAR-100, Flowers, Pets, Aircraft and Cars datasets.

### Preparation
Please download the pre-trained models, *e.g.*: 

* Download [DeiT-Tiny](https://drive.google.com/file/d/1RvhE2HucdWYHhKmPfHQW2A4EPpCHSYN_/view?usp=sharing) to `<BASE_FOLDER>/checkpoints/deit_tiny_300e.pth.tar`

* Download [MAE-Tiny](https://drive.google.com/file/d/1ZQYlvCPLZrJDqn2lp4GCIVL246WPqgEf/view?usp=sharing) to `<BASE_FOLDER>/checkpoints/mae_tiny_400e.pth.tar`

* Download [BEiT-Tiny](https://drive.google.com/file/d/13n_hY-DEheBAnheHWUoaYLS-cuCff1vX/view?usp=sharing) to `<BASE_FOLDER>/checkpoints/beit_tiny_400e.pth.tar`

* Download [BootMAE-Tiny](https://drive.google.com/file/d/11Oa9xH-6O1clrmWI5Vxyvk8DcHJ36c5P/view?usp=sharing) to `<BASE_FOLDER>/checkpoints/bootmae_tiny_400e.pth.tar`

* Download [MaskFeat-Tiny](https://drive.google.com/file/d/1Zbez38J0wn4qLR-ReoE6ey4LnCA2p2i6/view?usp=sharing) to `<BASE_FOLDER>/checkpoints/maskfeat_tiny_400e.pth.tar`

* Download [MoCov3-Tiny](https://drive.google.com/file/d/1RI0mU-PweAVIXs_hNOx-Xw3VRhN7w6un/view?usp=sharing) to `<BASE_FOLDER>/checkpoints/mocov3_tiny_300e.pth.tar`

The datasets will be downloaded automatically.

### Scripts
```bash
# 512 batch-sizes on 4 GPUs:
cd projects/eval_tools
ssl_train -b 512 -d 0-3 -e <epoch> -f finetuning_transfer_exp.py --amp \
--ckpt <checkpoint-path> --exp-options pretrain_exp_name=<pretrain-exp-name> \
dataset=<dataset> warmup_epochs=<warmup-epochs> save_folder_prefix=<prefix> \
lr=<lr> layer_decay=<decay>
```
- `<epoch>`: number of fine-tuning epochs;
- `<checkpoint-path>`: path to the checkpoint used as initialization, *e.g.*, `<BASE_FOLDER>/checkpoints/deit_tiny_300e.pth.tar`;
- `<dataset>`: `CIFAR10`, `CIFAR100`, `Aircraft`, `Flowers`, `Pets`, `Cars`
- `<warmup-epochs>`: warmup epochs
- `<pretrain-exp-name>&<prefix>`: we will save logs and ckpts to `<BASE_FOLDER>/outputs/<pretrain-exp-name>/<prefix>eval/`
- `<lr>`: learning rate
- `<decay>`: layer-wise lr decay

Please refer to [transfer.sh](scripts/transfer.sh) for detailed script examples.