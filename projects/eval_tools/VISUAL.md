## Visualization
Analysis tools to examine the layer representations and attention distance & entropy for the ViTs.

### Preparation
Please download the pre-trained models, *e.g.*: 

* Download [MAE-Tiny](https://drive.google.com/file/d/1ZQYlvCPLZrJDqn2lp4GCIVL246WPqgEf/view?usp=sharing) to `<BASE_FOLDER>/checkpoints/mae_tiny_400e.pth.tar`

* Download [DeiT-Tiny](https://drive.google.com/file/d/1RvhE2HucdWYHhKmPfHQW2A4EPpCHSYN_/view?usp=sharing) to `<BASE_FOLDER>/checkpoints/deit_tiny_300e.pth.tar`

### Layer Representation CKA Similarity 
```bash
cd projects/eval_tools
python run_vis_cmp_rep.py -b 128 --device cuda \
--save_name1 "MAE-Tiny" --ckpt1 "<BASE_FOLDER>/checkpoints/mae_tiny_400e.pth.tar" \
--exp-options1 pretrain_exp_name=mae_lite/mae_tiny_400e \
--save_name2 "DeiT-Tiny" --ckpt2 "<BASE_FOLDER>/checkpoints/deit_tiny_300e.pth.tar" \
--exp-options2 pretrain_exp_name=scratch/deit_tiny_300e
```
Then the CKA similarity heatmap between `MAE-Tiny` and `DeiT-Tiny` will be saved to `<BASE_FOLDER>/projects/eval_tools/results/rep_cmp/cka_MAE-Tiny+DeiT-Tiny.png`.

### Average Attention Distance & Entropy Analyses
```bash
cd projects/eval_tools
python run_vis_attn.py -b 128 --device cuda \
--save_name "MAE-Tiny" --ckpt "<BASE_FOLDER>/checkpoints/mae_tiny_400e.pth.tar" \
--exp-options pretrain_exp_name=mae_lite/mae_tiny_400e
```
Then the visualization results for `MAE-Tiny` will be saved to 
- `projects/eval_tools/results/attn/attnd_MAE-Tiny.png` (average attention distance);
- `projects/eval_tools/results/attn/attne_MAE-Tiny.png` (average attention entropy).
