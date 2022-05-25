# --------------------------------------------------------
# Copyright (c) 2022 Megvii, Inc. and its affiliates. All Rights Reserved.
# --------------------------------------------------------
import os
import time
from sys import stderr
from loguru import logger


def setup_logger(save_dir, distributed_rank=0, filename="log.txt", mode="a", timestamp=True):
    """setup logger for training and testing.
    Args:
        save_dir(str): location to save log file
        distributed_rank(int): device rank when multi-gpu environment
        mode(str): log file write mode, `append` or `override`. default is `a`.
        timestamp(bool): whether add timestamp to filename.
    Return:
        logger instance.
    """
    save_file = os.path.join(save_dir, filename)
    if timestamp:
        basename, extname = os.path.splitext(save_file)
        save_file = basename + time.strftime("-%Y-%m-%d-%H:%M:%S", time.localtime()) + extname

    if mode == "o" and os.path.exists(save_file):
        os.remove(save_file)
    format = f"[Rank #{distributed_rank}] | " + "{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}"
    if distributed_rank > 0:
        logger.remove()
        logger.add(stderr, format=format, level="WARNING")
    logger.add(
        save_file,
        format=format,
        filter="",
        level="INFO" if distributed_rank == 0 else "WARNING",
        enqueue=True,
    )

    return logger


def setup_tensorboard_logger(save_dir, distributed_rank=0, name="tb", timestamp=True):
    """setup tensorboard logger for training and testing.
    Args:
        save_dir(str): location to save log file
        distributed_rank(int): device rank when multi-gpu environment
        name(str): save folder
        timestamp(bool): whether add timestamp to `name`.
    Return:
        tensorboard logger instance for rank0, None for others.
    """
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        return None
    if distributed_rank == 0:
        # save_file = os.path.join(save_dir, name) if name else save_dir
        if timestamp:
            save_file = os.path.join(save_dir, name + time.strftime("-%Y-%m-%d-%H:%M:%S", time.localtime()))
        else:
            save_file = os.path.join(save_dir, name)
        writer = SummaryWriter(log_dir=save_file)
        return writer
    else:
        return None
