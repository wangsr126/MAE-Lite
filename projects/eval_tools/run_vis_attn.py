# --------------------------------------------------------
# Copyright (c) 2022 Megvii, Inc. and its affiliates. All Rights Reserved.
# --------------------------------------------------------
import os, sys
import argparse
import importlib
from tqdm import tqdm
import torch
import torch.backends.cudnn as cudnn
from mae_lite.utils import DictAction
from mae_lite.utils.log import setup_logger

import finetuning_exp
from visualize import AttnAnalyzer
from util.forward_hook import AttnCatcher


def get_args():
    parser = argparse.ArgumentParser('attention visualization script', add_help=False)
    parser.add_argument("-b", "--batch-size", type=int, default=16, help="batch size")
    parser.add_argument("-n", "--num", type=int, default=1024, help="total numbers")
    parser.add_argument('--save_name', type=str, default='mae', help='save name')
    parser.add_argument('--device', default='cpu',
                        help='device to use for training / testing')
    parser.add_argument(
        "-f",
        "--exp_file",
        default=finetuning_exp.__file__,
        type=str,
        help="pls input your expriment description file",
    )
    parser.add_argument("--ckpt", default=None, type=str, help="checkpoint path for visualization")
    parser.add_argument(
        "--exp-options",
        nargs="+",
        action=DictAction,
        help="override some settings in the exp, the key-value pair in xxx=yyy format will be merged into exp. "
        'If the value to be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        "Note that the quotation marks are necessary and that no white space is allowed.",
    )

    return parser.parse_args()


def main(args):
    print(args)

    device = torch.device(args.device)
    cudnn.benchmark = True

    sys.path.insert(0, os.path.dirname(args.exp_file))
    current_exp = importlib.import_module(os.path.basename(args.exp_file).split(".")[0])

    exp = current_exp.Exp(args.batch_size, 1)
    update_cfg_msg = exp.update(args.exp_options)
    pretrained_file_name = os.path.join(exp.output_dir, exp.pretrain_exp_name)
    file_name = os.path.join(pretrained_file_name, "vis")
    logger = setup_logger(file_name, distributed_rank=0, filename="vis_log.txt", mode="a")
    logger.opt(ansi=True).info(
        "<yellow>List of override configs</yellow>:\n<blue>{}</blue>".format(update_cfg_msg)
    )
    model = exp.get_model()

    model.to(device)

    if args.ckpt and os.path.isfile(args.ckpt):
        ckpt_path = args.ckpt
    else:
        ckpt_path = os.path.join(pretrained_file_name, "last_epoch_ckpt.pth.tar")
    msg = exp.set_model_weights(ckpt_path)
    logger.warning("Model params {} are not loaded".format(msg.missing_keys))
    logger.warning("State-dict params {} are not used".format(msg.unexpected_keys))

    model.eval()
    catcher = AttnCatcher(model.model, list(range(1, model.model.depth+1)))

    data_loader = exp.get_data_loader()["eval"]

    attn_analyzer = AttnAnalyzer(model.model.depth, model.model.num_heads)
    total_length = 0
    pbar = tqdm(range(len(data_loader)))
    with torch.no_grad():
        for _, (imgs, _) in zip(pbar, data_loader):
            imgs = imgs.to(device)
            tokens = model.model.forward_features(imgs)
            attns = catcher.get_features()
            attn_analyzer.append(attns)
            total_length += tokens.size(0)
            if total_length > args.num:
                break

    # plot attn
    attn_analyzer.plot_box_attn(os.path.join("results/attn/", "attn_{}".format(args.save_name)), title=args.save_name)


if __name__ == '__main__':
    opts = get_args()
    main(opts)
