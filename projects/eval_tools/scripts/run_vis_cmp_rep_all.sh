cd ../

# DeiT-Tiny vs. DeiT-Tiny
python run_vis_cmp_rep.py -b 128 --device cuda \
--save_name1 "DeiT-Tiny" --ckpt1 "../../checkpoints/deit_tiny_300e.pth.tar" \
--exp-options1 pretrain_exp_name=scratch/deit_tiny_300e \
--save_name2 "DeiT-Tiny" --ckpt2 "../../checkpoints/deit_tiny_300e.pth.tar" \
--exp-options2 pretrain_exp_name=scratch/deit_tiny_300e
# MAE-Tiny vs. DeiT-Tiny
python run_vis_cmp_rep.py -b 128 --device cuda \
--save_name1 "MAE-Tiny" --ckpt1 "../../checkpoints/mae_tiny_400e.pth.tar" \
--exp-options1 pretrain_exp_name=mae_lite/mae_tiny_400e \
--save_name2 "DeiT-Tiny" --ckpt2 "../../checkpoints/deit_tiny_300e.pth.tar" \
--exp-options pretrain_exp_name=scratch/deit_tiny_300e
# MoCov3-Tiny vs. DeiT-Tiny
python run_vis_cmp_rep.py -b 128 --device cuda \
--save_name1 "MoCov3-Tiny" --ckpt1 "../../checkpoints/mocov3_tiny_400e.pth.tar" \
--exp-options1 pretrain_exp_name=mae_lite/mae_tiny_400e global_pool=False weights_prefix="base_encoder" \
--save_name2 "DeiT-Tiny" --ckpt2 "../../checkpoints/deit_tiny_300e.pth.tar" \
--exp-options pretrain_exp_name=scratch/deit_tiny_300e
# MAE-Base vs. DeiT-Tiny
python run_vis_cmp_rep.py -b 64 --device cuda \
--save_name1 "MAE-Base" --ckpt1 "../../checkpoints/mae_base_1600e.pth.tar" \
--exp-options1 encoder_arch=vit_base_patch16 pretrain_exp_name=mae_lite/mae_base_1600e weights_prefix="" \
--save_name2 "DeiT-Tiny" --ckpt2 "../../checkpoints/deit_tiny_300e.pth.tar" \
--exp-options pretrain_exp_name=scratch/deit_tiny_300e
# D-MAE-Tiny vs. DeiT-Tiny
python run_vis_cmp_rep.py -b 128 --device cuda \
--save_name1 "D-MAE-Tiny" --ckpt1 "../../checkpoints/mae_tiny_distill_400e.pth.tar" \
--exp-options1 pretrain_exp_name=mae_lite/mae_tiny_distill_400e \
--save_name2 "DeiT-Tiny" --ckpt2 "../../checkpoints/deit_tiny_300e.pth.tar" \
--exp-options pretrain_exp_name=scratch/deit_tiny_300e
