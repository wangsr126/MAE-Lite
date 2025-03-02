## Transfer to COCO Detection Tasks
Transfer evaluation of the pre-trained models on COCO detection tasks based on [ViTDet](https://github.com/facebookresearch/detectron2/blob/main/projects/ViTDet).

### Preparation
Please follow the instructions to install [detectron2](https://github.com/facebookresearch/detectron2) and prepare COCO dataset.

Then download the pre-trained models, *e.g.*: 

* Download [DeiT-Tiny](https://drive.google.com/file/d/1RvhE2HucdWYHhKmPfHQW2A4EPpCHSYN_/view?usp=sharing) to `<BASE_FOLDER>/checkpoints/deit_tiny_300e.pth.tar`

* Download [MAE-Tiny](https://drive.google.com/file/d/1ZQYlvCPLZrJDqn2lp4GCIVL246WPqgEf/view?usp=sharing) to `<BASE_FOLDER>/checkpoints/mae_tiny_400e.pth.tar`

* Download [MoCov3-Tiny](https://drive.google.com/file/d/1RI0mU-PweAVIXs_hNOx-Xw3VRhN7w6un/view?usp=sharing) to `<BASE_FOLDER>/checkpoints/mocov3_tiny_400e.pth.tar`

And convert them:
```bash
python projects/eval_tools/det/convert_to_d2.py checkpoints/deit_tiny_300e.pth.tar checkpoints/deit_tiny_300e.pth -wp 'module.model.'
python projects/eval_tools/det/convert_to_d2.py checkpoints/mae_tiny_400e.pth.tar checkpoints/mae_tiny_400e.pth -wp 'module.model.'
python projects/eval_tools/det/convert_to_d2.py checkpoints/mocov3_tiny_400e.pth.tar checkpoints/mocov3_tiny_400e.pth -wp 'module.base_encoder.'
```

### Fine-Tuning on COCO
```bash
# on 8 GPUs:
cd projects/eval_tools/det
<PATH_TO_DETECTRON2>/tools/lazyconfig_train_net.py --config-file mask_rcnn_vitdet_t_100ep.py
```

## Main Results
By default, the configs use `image_size=640`.
|Config|Initialization | AP<sup>bb</sup><br> | AP<sup>m</sup><br> | ckpt | 
|---|---|---|---|---|
|[mask_rcnn_vitdet_t_100ep.py](mask_rcnn_vitdet_t_100ep.py)| [DeiT-Tiny](https://drive.google.com/file/d/1RvhE2HucdWYHhKmPfHQW2A4EPpCHSYN_/view?usp=sharing) | 40.4 | 35.5 | [link](https://drive.google.com/file/d/13EGdnjrvXsadnvnMETB2bwBs1iB_g8ZQ/view?usp=sharing) |
|[mask_rcnn_vitdet_t_100ep_mae.py](mask_rcnn_vitdet_t_100ep_mae.py)| [MAE-Tiny](https://drive.google.com/file/d/1ZQYlvCPLZrJDqn2lp4GCIVL246WPqgEf/view?usp=sharing) | 39.9 | 35.4 | [link](https://drive.google.com/file/d/18ttBrHFEje0LFDzF7RPVV767mh4oU-qp/view?usp=sharing) |
|[mask_rcnn_vitdet_t_100ep_mae_distill.py](mask_rcnn_vitdet_t_100ep_mae_distill.py)| [D-MAE-Tiny](https://drive.google.com/file/d/1OCDMUEdcPhwoCPWGN0kahsHST7tbQmFe/view?usp=sharing) | 42.3 | 37.3 | [link](https://drive.google.com/file/d/12iAH_hfTd57E-4XQzcWIPjWGp4V5Whq-/view?usp=sharing) |
|[mask_rcnn_vitdet_t_100ep_mae_d2_distill.py](mask_rcnn_vitdet_t_100ep_mae_d2_distill.py)| [D²-MAE-Tiny](https://drive.google.com/file/d/1gOKB8lSQ3IlOLNbi5Uc7saZO2mMClXNU/view?usp=sharing) | 42.5 | 37.5 | [link](https://drive.google.com/file/d/1gWNP2Z5mCnGHWSQsOmaQDGspkByMzjH2/view?usp=sharing) |
|[mask_rcnn_vitdet_t_100ep_mocov3.py](mask_rcnn_vitdet_t_100ep_mocov3.py)| [MoCov3-Tiny](https://drive.google.com/file/d/1RI0mU-PweAVIXs_hNOx-Xw3VRhN7w6un/view?usp=sharing) | 39.7 | 35.1 | [link](https://drive.google.com/file/d/1fdyj8n8BuyzCx1bIhhc7dF4XUN54myHs/view?usp=sharing) |