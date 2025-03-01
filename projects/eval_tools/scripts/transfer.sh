cd ../
# -----------------------------------------------CIFAR100--------------------------------------------------
# DeiT-Tiny
ssl_train -b 512 -d 0-1 -f finetuning_transfer_exp.py -e 50 --amp \
--ckpt "../../checkpoints/deit_tiny_300e.pth.tar" --exp-options pretrain_exp_name=scratch/deit_tiny_300e \
dataset=CIFAR100 save_folder_prefix='cifar_' num_classes=100 warmup_epochs=10 lr=0.1 layer_decay=0.75
# MAE-Tiny
ssl_train -b 512 -d 0-1 -f finetuning_transfer_exp.py -e 50 --amp \
--ckpt "../../checkpoints/mae_tiny_400e.pth.tar" --exp-options pretrain_exp_name=mae_lite/mae_tiny_400e \
dataset=CIFAR100 save_folder_prefix='cifar_' num_classes=100 warmup_epochs=10 lr=0.1 layer_decay=0.75
# BEiT-Tiny
ssl_train -b 512 -d 0-1 -f finetuning_transfer_exp.py -e 50 --amp \
--ckpt "../../checkpoints/beit_tiny_400e.pth.tar" --exp-options pretrain_exp_name=beit/beit_tiny_400e weights_prefix="" encoder_arch="vit_tiny_patch16_rpe" \
dataset=CIFAR100 save_folder_prefix='cifar_' num_classes=100 warmup_epochs=10 lr=0.1 layer_decay=0.75 use_rel_pos_bias=False init_values=0.1
# BootMAE-Tiny
ssl_train -b 512 -d 0-1 -f finetuning_transfer_exp.py -e 50 --amp \
--ckpt "../../checkpoints/bootmae_tiny_400e.pth.tar" --exp-options pretrain_exp_name=bootmae/bootmae_tiny_400e weights_prefix="" encoder_arch="vit_tiny_patch16_rpe" \
dataset=CIFAR100 save_folder_prefix='cifar_' num_classes=100 warmup_epochs=10 lr=0.1 layer_decay=0.75 use_rel_pos_bias=False 
# MaskFeat-Tiny
ssl_train -b 512 -d 0-1 -f finetuning_transfer_exp.py -e 50 --amp \
--ckpt "../../checkpoints/maskfeat_tiny_400e.pth.tar" --exp-options pretrain_exp_name=maskfeat/maskfeat_tiny_400e weights_prefix="" \
dataset=CIFAR100 save_folder_prefix='cifar_' num_classes=100 warmup_epochs=10 lr=0.1 layer_decay=0.75
# MoCov3-Tiny
ssl_train -b 512 -d 0-1 -f finetuning_transfer_exp.py -e 50 --amp \
--ckpt "../../checkpoints/mocov3_tiny_400e.pth.tar" --exp-options pretrain_exp_name=mocov3/mocov3_tiny_400e weights_prefix="base_encoder" \
dataset=CIFAR100 save_folder_prefix='cifar_' num_classes=100 warmup_epochs=10 lr=0.1 layer_decay=0.75
