cd ../
# scratch
python run_vis_attn.py -b 128 --device cuda --save_name "Init" --exp-options \
pretrain_exp_name=scratch
# deit
python run_vis_attn.py -b 128 --device cuda --save_name "DeiT-Tiny" --ckpt "../../checkpoints/deit_tiny_300e.pth.tar" \
--exp-options pretrain_exp_name=scratch/deit_tiny_300e
# mae
python run_vis_attn.py -b 128 --device cuda --save_name "MAE-Tiny" --ckpt "../../checkpoints/mae_tiny_400e.pth.tar" \
--exp-options pretrain_exp_name=mae_lite/mae_tiny_400e
# mae-distill
python run_vis_attn.py -b 128 --device cuda --save_name "D-MAE-Tiny" --ckpt "../../checkpoints/mae_tiny_distill_400e.pth.tar" \
--exp-options pretrain_exp_name=mae_lite/mae_tiny_distill_400e
# mae-ft
python run_vis_attn.py -b 128 --device cuda --save_name "MAE-Tiny-FT" --ckpt "../../checkpoints/mae_tiny_400e_ft_300e.pth.tar" \
--exp-options pretrain_exp_name=mae_lite/mae_tiny_400e/ft_300e
# mae-base
python run_vis_attn.py -b 64 --device cuda --save_name "MAE-Base" --ckpt "../../checkpoints/mae_base_1600e.pth.tar" \
--exp-options encoder_arch=vit_base_patch16 weights_prefix="" pretrain_exp_name=mae_lite/mae_base_1600e
# mocov3
python run_vis_attn.py -b 128 --device cuda --save_name "MoCov3-Tiny" --ckpt "../../checkpoints/mocov3_tiny_400e.pth.tar" \
--exp-options weights_prefix="base_encoder" pretrain_exp_name=mocov3/mocov3_tiny_400e
