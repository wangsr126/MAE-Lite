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
from visualize import RepCmpAnalyzer
from util.forward_hook import RepCatcher


def get_args():
    parser = argparse.ArgumentParser('CKA similarity visualization script', add_help=False)
    parser.add_argument("-b", "--batch-size", type=int, default=16, help="batch size")
    parser.add_argument("-n", "--num", type=int, default=1024, help="total numbers")
    parser.add_argument('--save_name1', type=str, default='mae', help='save name')
    parser.add_argument('--save_name2', type=str, default='mae', help='save name')
    parser.add_argument('--device', default='cpu',
                        help='device to use for training / testing')
    parser.add_argument(
        "-f",
        "--exp_file",
        default=finetuning_exp.__file__,
        type=str,
        help="pls input your expriment description file",
    )
    parser.add_argument("--ckpt1", default=None, type=str, help="checkpoint path for visualization")
    parser.add_argument(
        "--exp-options1",
        nargs="+",
        action=DictAction,
        help="override some settings in the exp, the key-value pair in xxx=yyy format will be merged into exp. "
        'If the value to be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        "Note that the quotation marks are necessary and that no white space is allowed.",
    )
    parser.add_argument("--ckpt2", default=None, type=str, help="checkpoint path for visualization")
    parser.add_argument(
        "--exp-options2",
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

    exp1 = current_exp.Exp(args.batch_size, 1)
    update_cfg_msg = exp1.update(args.exp_options1)
    pretrained_file_name1 = os.path.join(exp1.output_dir, exp1.pretrain_exp_name)
    file_name = os.path.join(pretrained_file_name1, "vis")
    logger = setup_logger(file_name, distributed_rank=0, filename="vis_log.txt", mode="a")
    logger.opt(ansi=True).info(
        "<yellow>List of override configs</yellow>:\n<blue>{}</blue>".format(update_cfg_msg)
    )
    model1 = exp1.get_model()
    model1.to(device)
    if args.ckpt1 and os.path.isfile(args.ckpt1):
        ckpt_path1 = args.ckpt1
    else:
        ckpt_path1 = os.path.join(pretrained_file_name1, "last_epoch_ckpt.pth.tar")
    msg = exp1.set_model_weights(ckpt_path1)
    logger.warning("Model params {} are not loaded".format(msg.missing_keys))
    logger.warning("State-dict params {} are not used".format(msg.unexpected_keys))
    model1.eval()
    catcher1 = RepCatcher(model1.model, list(range(model1.model.depth+1)))

    exp2 = current_exp.Exp(args.batch_size, 1)
    update_cfg_msg = exp2.update(args.exp_options2)
    pretrained_file_name2 = os.path.join(exp2.output_dir, exp2.pretrain_exp_name)
    logger.opt(ansi=True).info(
        "<yellow>List of override configs</yellow>:\n<blue>{}</blue>".format(update_cfg_msg)
    )
    model2 = exp2.get_model()
    model2.to(device)
    if args.ckpt2 and os.path.isfile(args.ckpt2):
        ckpt_path2 = args.ckpt2
    else:
        ckpt_path2 = os.path.join(pretrained_file_name2, "last_epoch_ckpt.pth.tar")
    msg = exp2.set_model_weights(ckpt_path2)
    logger.warning("Model params {} are not loaded".format(msg.missing_keys))
    logger.warning("State-dict params {} are not used".format(msg.unexpected_keys))
    model2.eval()
    catcher2 = RepCatcher(model2.model, list(range(model2.model.depth+1)))

    data_loader = exp1.get_data_loader()["eval"]

    # add patch embedding layer
    rep_cmp_analyzer = RepCmpAnalyzer(model1.model.depth+1, model2.model.depth+1)
    pbar = tqdm(range(len(data_loader)))
    total_length = 0
    with torch.no_grad():
        for _, (imgs, _) in zip(pbar, data_loader):
            imgs = imgs.to(device)
            tokens1 = model1.model.forward_features(imgs)
            tokens2 = model2.model.forward_features(imgs)
            hiddens1 = catcher1.get_features()
            hiddens2 = catcher2.get_features()
            rep_cmp_analyzer.append(hiddens1, hiddens2)
            total_length += tokens1.size(0)
            if total_length > args.num:
                break

    # plot cka
    rep_cmp_analyzer.plot_rep_cmp(
        os.path.join("results/rep_cmp/", "cka_{}+{}".format(args.save_name1, args.save_name2)),
        args.save_name1,
        args.save_name2
    )


if __name__ == "__main__":
    opts = get_args()
    main(opts)
