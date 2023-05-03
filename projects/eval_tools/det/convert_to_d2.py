# --------------------------------------------------------
# Copyright (c) 2022 Megvii, Inc. and its affiliates. All Rights Reserved.
# --------------------------------------------------------
import pickle as pkl
import torch
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="This script convert pretrained model to detectron2")
    parser.add_argument("checkpoint", help="checkpoint file path")
    parser.add_argument("output", type=str, help="destination file path")
    parser.add_argument("-t", "--type", type=str, default="naive", help="type of model architecture, resnet / naive")
    parser.add_argument("-wp", "--weights_prefix", type=str, default="module.model.", help="prefix of weights names")
    args = parser.parse_args()
    return args


def convert_resnet(ckpt, prefix, black_list=("fc",)):
    # extract backbone
    state_dict = dict()
    for key, value in ckpt["model"].items():
        if prefix in key:
            new_key = key.replace(prefix, "")
            if not any(map(new_key.startswith, black_list)):
                state_dict[new_key] = value
    assert state_dict, "No weights with prefix `{}`!".format(prefix)

    # convert to detectron
    new_model = {}
    for k, v in state_dict.items():
        old_k = k
        if "layer" not in k:
            k = "stem." + k
        for t in [1, 2, 3, 4]:
            k = k.replace("layer{}".format(t), "res{}".format(t + 1))
        for t in [1, 2, 3]:
            k = k.replace("bn{}".format(t), "conv{}.norm".format(t))
        k = k.replace("downsample.0", "shortcut")
        k = k.replace("downsample.1", "shortcut.norm")
        print(old_k, "->", k)
        new_model[k] = v
    return new_model


def convert_naive(ckpt, prefix, black_list=("fc",)):
    # extract backbone
    state_dict = dict()
    for key, value in ckpt["model"].items():
        if prefix in key:
            new_key = key.replace(prefix, "")
            if not any(map(new_key.startswith, black_list)):
                state_dict[new_key] = value
            state_dict[new_key] = value
    assert state_dict, "No weights with prefix `{}`!".format(prefix)
    new_model = {}
    for k, v in state_dict.items():
        new_k = k
        print(k, "->", new_k)
        new_model[new_k] = v
    return new_model


CONVERT_FUNC_DICT = {
    "resnet": convert_resnet,
    "naive": convert_naive,
}


def main():
    args = parse_args()
    ckpt_path = args.checkpoint
    assert ckpt_path.endswith(".pth") or ckpt_path.endswith(".pth.tar")
    ckpt = torch.load(ckpt_path, map_location=torch.device("cpu"))

    epoch = ckpt["start_epoch"] if "start_epoch" in ckpt else -1
    print("Start to convert '{}' (epoch={})".format(ckpt_path, epoch))

    new_model = CONVERT_FUNC_DICT[args.type](ckpt, args.weights_prefix)

    output_path = args.output
    if output_path.endswith(".pth"):
        res = {"model": new_model}
        torch.save(res, args.output)
    elif output_path.endswith(".pkl"):
        res = {"model": new_model, "__author__": "MegSSL", "matching_heuristics": True}
        with open(output_path, "wb") as f:
            pkl.dump(res, f)
    else:
        raise NameError
    print("dump converted checkpoints to {}".format(args.output))


if __name__ == "__main__":
    main()
