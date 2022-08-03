import argparse
import importlib
import os
import subprocess
import sys
import time
import torch

import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from mae_lite.utils import (
    DictAction,
    AvgMeter,
    accuracy,
    setup_logger,
    collect_env_info,
    random_seed,
)
from mae_lite.utils.torch_dist import parse_devices, configure_nccl, all_reduce_mean, synchronize

from mae_lite.exps import timm_imagenet_exp


def get_arg_parser():
    parser = argparse.ArgumentParser("Classification Evaluation")
    # distributed
    parser.add_argument("--dist-backend", default="nccl", type=str, help="distributed backend")
    parser.add_argument("--dist-url", default=None, type=str, help="url used to set up distributed training")
    parser.add_argument("-b", "--batch-size", type=int, default=1024, help="batch size")
    parser.add_argument("-d", "--devices", default="0-7", type=str, help="device for training")
    parser.add_argument(
        "-f",
        "--exp_file",
        default=timm_imagenet_exp.__file__,
        type=str,
        help="pls input your expriment description file",
    )
    parser.add_argument("--ckpt", default=None, type=str, help="checkpoint path for evaluation")
    parser.add_argument(
        "--exp-options",
        nargs="+",
        action=DictAction,
        help="override some settings in the exp, the key-value pair in xxx=yyy format will be merged into exp. "
        'If the value to be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        "Note that the quotation marks are necessary and that no white space is allowed.",
    )
    args = parser.parse_args()
    return args


def main():
    args = get_arg_parser()
    args.devices = parse_devices(args.devices)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.devices
    nr_gpu = len(args.devices.split(","))

    nr_machine = int(os.getenv("MACHINE_TOTAL", "1"))
    if nr_gpu > 1:
        args.world_size = nr_gpu * nr_machine
        processes = []
        for rank in range(nr_gpu):
            p = mp.Process(target=main_worker, args=(rank, nr_gpu, args))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
    else:
        main_worker(0, nr_gpu, args)


def main_worker(gpu, nr_gpu, args):
    current_exp_name = os.path.basename(args.exp_file).split(".")[0]
    # ------------ set environment variables for distributed training ------------------------------------- #
    configure_nccl()
    rank = gpu
    if nr_gpu > 1:
        rank += int(os.getenv("MACHINE_RANK", "0")) * nr_gpu

        if args.dist_url is None:
            master_ip = subprocess.check_output(["hostname", "--fqdn"]).decode("utf-8")
            master_ip = str(master_ip).strip()
            args.dist_url = "tcp://{}:23456".format(master_ip)

            # ------------------------hack for multi-machine training -------------------- #
            if args.world_size > 8:
                ip_add_file = "./" + current_exp_name + "ip_add.txt"
                if rank == 0:
                    with open(ip_add_file, "w") as ip_add:
                        ip_add.write(args.dist_url)
                else:
                    while not os.path.exists(ip_add_file):
                        time.sleep(0.5)

                    with open(ip_add_file, "r") as ip_add:
                        dist_url = ip_add.readline()
                    args.dist_url = dist_url
        else:
            args.dist_url = "tcp://{}:23456".format(args.dist_url)

        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=rank,
        )
        print("Rank {} initialization finished.".format(rank))
        synchronize()

        if rank == 0:
            if os.path.exists("./" + current_exp_name + "ip_add.txt"):
                os.remove("./" + current_exp_name + "ip_add.txt")

    sys.path.insert(0, os.path.dirname(args.exp_file))
    current_exp = importlib.import_module(os.path.basename(args.exp_file).split(".")[0])

    exp = current_exp.Exp(args.batch_size)
    update_cfg_msg = exp.update(args.exp_options)

    if exp.seed is not None:
        random_seed(exp.seed, rank)

    file_name = os.path.join(exp.output_dir, exp.exp_name)

    if rank == 0:
        os.makedirs(file_name, exist_ok=True)

    logger = setup_logger(file_name, distributed_rank=rank, filename="eval_log.txt", mode="a")
    if rank == 0:
        logger.info("gpuid: {}, args: <>{}".format(rank, args))
        logger.opt(ansi=True).info(
            "<yellow>Used experiment configs</yellow>:\n<blue>{}</blue>".format(exp.get_cfg_as_str())
        )
        if update_cfg_msg:
            logger.opt(ansi=True).info(
                "<yellow>List of override configs</yellow>:\n<blue>{}</blue>".format(update_cfg_msg)
            )
        logger.opt(ansi=True).info("<yellow>Environment info:</yellow>\n<blue>{}</blue>".format(collect_env_info()))

    data_loader = exp.get_data_loader()
    eval_loader = data_loader["eval"]
    model = exp.get_model()
    if rank == 0:
        logger.info("Illustration of model strcutures:\n{}".format(str(model)))
    torch.cuda.set_device(gpu)
    model.cuda(gpu)
    if nr_gpu > 1:
        model = DDP(model, device_ids=[gpu])

    #  ------------------------------------------- load ckpt ------------------------------------ #
    # specify the path of the ckeckpoint for evaluation.
    if args.ckpt and os.path.isfile(args.ckpt):
        ckpt_path = args.ckpt 
    else:
        # Automaticly load the lastest checkpoint.
        ckpt_path = os.path.join(file_name, "last_epoch_best_ckpt.pth.tar")
        if os.path.isfile(ckpt_path):
            ckpt_path = os.path.join(file_name, "last_epoch_ckpt.pth.tar")
            assert os.path.isfile(ckpt_path), "Failed to load ckpt from '{}'".format(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    msg = model.load_state_dict(ckpt["model"])
    if rank == 0:
        logger.warning("Model params {} are not loaded".format(msg.missing_keys))
        logger.warning("State-dict params {} are not used".format(msg.unexpected_keys))

    model.eval()
    eval_top1, eval_top5 = run_eval(model, eval_loader)
    if rank == 0:
        logger.info("Evaluation of experiment: {} is done.".format(exp.exp_name))
        logger.info(
            "\tTop1:{:.3f}, Top5:{:.3f}".format(eval_top1, eval_top5)
        )
    logger.stop()


def run_eval(model, eval_loader):

    top1 = AvgMeter()
    top5 = AvgMeter()

    with torch.no_grad():
        pbar = tqdm(range(len(eval_loader)))
        for _, (inp, target) in zip(pbar, eval_loader):
            inp = inp.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            logits = model(inp)
            acc1, acc5 = accuracy(logits, target, (1, 5))
            acc1, acc5 = all_reduce_mean(acc1), all_reduce_mean(acc5)
            top1.update(acc1.item(), inp.size(0))
            top5.update(acc5.item(), inp.size(0))
    return top1.avg, top5.avg


if __name__ == "__main__":
    main()
