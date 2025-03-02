## Transfer to LaSOT Tracking Tasks
Transfer evaluation of the pre-trained models on LaSOT tracking tasks based on [OSTrack](https://github.com/botaoye/OSTrack.git).

### Preparation
Please follow the instructions to install [OSTrack](https://github.com/botaoye/OSTrack.git) and prepare datasets.

Then download the pre-trained models and modified the path in [configs](experiments/ostrack/README.md), *e.g.*: 

* Download [DeiT-Tiny](https://drive.google.com/file/d/1RvhE2HucdWYHhKmPfHQW2A4EPpCHSYN_/view?usp=sharing) to `<BASE_FOLDER>/checkpoints/deit_tiny_300e.pth.tar`

* Download [MAE-Distill-Tiny](https://drive.google.com/file/d/1OCDMUEdcPhwoCPWGN0kahsHST7tbQmFe/view?usp=sharing) to `<BASE_FOLDER>/checkpoints/deit_tiny_300e.pth.tar`

* Download [MAE-Distill-D²-Tiny](https://drive.google.com/file/d/1gOKB8lSQ3IlOLNbi5Uc7saZO2mMClXNU/view?usp=sharing) to `<BASE_FOLDER>/checkpoints/mae_tiny_distill_d2_400e.pth.tar`

### Fine-Tuning on LaSOT
```bash
# on 4 GPUs:
cd projects/eval_tools/track
python tracking/train.py --script ostrack --config vit_tiny_256_ce_32x4_ep300 --save_dir ./output --mode multiple --nproc_per_node 4 --use_wandb 1
```

### Evaluation
```bash
# on 4 GPUs:
cd projects/eval_tools/track
python tracking/test.py ostrack vitb_384_mae_ce_32x4_ep300 --dataset lasot --threads 16 --num_gpus 4
python tracking/analysis_results.py # need to modify tracker configs and names
```

## Main Results
By default, the configs use `search_size=256`.
|Config|Initialization | AUC | Speed (CPU) | ckpt | 
|---|---|---|---|---|
|[vit_tiny_256_ce_32x4_ep300.yaml](experiments/ostrack/vit_tiny_256_ce_32x4_ep300.yaml)| [DeiT-Tiny](https://drive.google.com/file/d/1RvhE2HucdWYHhKmPfHQW2A4EPpCHSYN_/view?usp=sharing) | 64.1 | 41 | [link](https://drive.google.com/file/d/1Bd6IRwKB-l0nkRH4VdzFJO24QtKFLug_/view?usp=sharing) |
|[vit_tiny_256_mae_distill_ce_32x4_ep300.yaml](experiments/ostrack/vit_tiny_256_mae_distill_ce_32x4_ep300.yaml)| [MAE-Distill-Tiny](https://drive.google.com/file/d/1OCDMUEdcPhwoCPWGN0kahsHST7tbQmFe/view?usp=sharing) | 65.8 | 41 | [link](https://drive.google.com/file/d/1viK0MnP1kHnZYtcbZjvaeeVV5lb9MvoH/view?usp=sharing) |
|[vit_tiny_256_mae_distill_d2_ce_32x4_ep300.yaml](experiments/ostrack/vit_tiny_256_mae_distill_d2_ce_32x4_ep300.yaml)| [MAE-Distill-D²-Tiny](https://drive.google.com/file/d/1gOKB8lSQ3IlOLNbi5Uc7saZO2mMClXNU/view?usp=sharing) | 66.1 | 41 | [link](https://drive.google.com/file/d/19LfcEc0cbp15MglcfG9Rpp24aYV-bRux/view?usp=sharing) |
