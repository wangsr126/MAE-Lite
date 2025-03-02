# --------------------------------------------------------
# Copyright (c) 2022 Megvii, Inc. and its affiliates. All Rights Reserved.
# ---------------------------------------------------------
""" 
LRSeheduler.
"""
import math
import torch
from typing import List
from functools import partial
from .registry import LRSCHEDULERS


__all__ = ["LRScheduler", "CosineLRScheduler", "WarmCosineLRScheduler", "StepLRScheduler"]


class _Scheduler:
    def __init__(self, lr, total_steps):
        self.lr = lr
        self.total_steps = total_steps
        self._get_lr = self._get_lr_func()

    def _get_lr_func(self):
        def _get_lr(count):
            raise NotImplementedError
        return _get_lr

    def __call__(self, count):
        return self._get_lr(count)


@LRSCHEDULERS.register(name="base")
class LRScheduler:
    """Similar to torch.optimizer._LRScheduler, but
    the current_lr is calculated according to current_step;
    scheduler takes the full control on lr update;
    """

    def __init__(self, optimizer: torch.optim.Optimizer, scheduler: _Scheduler, interval: int = 1):
        """
        Args:
            optimizer: See ``torch.optim.lr_scheduler._LRScheduler``.
            scheduler: a _Scheduler that defines the scheduler on params of the optimizer or
                a tuple of _Scheduler for the params groups of the optimizer.
            interval: update interval.
        """
        if isinstance(scheduler, _Scheduler):
            scheduler = (scheduler,) * len(optimizer.param_groups)
        if not all([isinstance(s, _Scheduler) for s in scheduler]):
            raise ValueError(
                "_LRScheduler(scheduler=) must be an instance or a tuple of instances of "
                f"_Scheduler. Got {scheduler} instead."
            )
        self._scheduler = scheduler
        assert len(self._scheduler) == len(optimizer.param_groups), "Please provide "
        "scheduler with the same number as ``optimizer.param_groups``, or just one scheduler "
        "shared among all ``optimizer.param_groups``."

        self.optimizer = optimizer
        self.base_lrs = [sched.lr for sched in self._scheduler]
        self.interval = interval
        self.last_count = -1

    def state_dict(self):
        # _scheduler is stateless.
        return {"last_count": self.last_count}

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_last_lr(self):
        """Return last computed learning rate by current scheduler."""
        return [group["lr"] for group in self.optimizer.param_groups]

    def get_last_lr_str(self):
        lrs = self.get_last_lr()
        if len(set(lrs)) == 1:
            return "{:.4f}".format(lrs[0])
        if len(lrs) < 3:
            lr_str = "-".join(["{:.4f}".format(lr) for lr in lrs])
        else:
            lr_str = "{:.4f}(max)-{:.4f}(min)".format(max(lrs), min(lrs))
        return lr_str

    def get_lr(self, count) -> List[float]:
        return [sched(count) for sched in self._scheduler]

    def step(self, count):
        count, inner_count = divmod(count, self.interval)
        if inner_count == 0:
            values = self.get_lr(count)
            for param_group, lr in zip(self.optimizer.param_groups, values):
                param_group["lr"] = lr
            self.last_count = count


class _CosineScheduler(_Scheduler):
    def __init__(self, lr, total_steps, end_lr=0.0):
        self.end_lr = end_lr
        super(_CosineScheduler, self).__init__(lr, total_steps)

    def _get_lr_func(self):
        def cos_lr(lr, total_steps, count):
            """Cosine learning rate"""
            lr *= 0.5 * (1.0 + math.cos(math.pi * count / total_steps))
            return lr

        def cos_w_end_lr(lr, total_steps, end_lr, count):
            """Cosine learning rate with end lr"""
            q = 0.5 * (1.0 + math.cos(math.pi * count / total_steps))
            lr = lr * q + end_lr * (1 - q)
            return lr

        end_lr = self.end_lr
        if end_lr > 0:
            lr_func = partial(cos_w_end_lr, self.lr, self.total_steps, end_lr)
        else:
            lr_func = partial(cos_lr, self.lr, self.total_steps)
        return lr_func


@LRSCHEDULERS.register(name="cos")
class CosineLRScheduler(LRScheduler):
    def __init__(self, optimizer, lr, total_steps, end_lr=0.0):
        scheduler = _CosineScheduler(lr, total_steps, end_lr=end_lr)
        super(CosineLRScheduler, self).__init__(optimizer, scheduler)


class _WarmCosineScheduler(_CosineScheduler):
    def __init__(self, lr, total_steps, warmup_steps=0.0, warmup_lr_start=1e-6, end_lr=0.0):
        self.warmup_steps = warmup_steps
        self.warmup_lr_start = warmup_lr_start
        super(_WarmCosineScheduler, self).__init__(lr, total_steps, end_lr)

    def _get_lr_func(self):
        def warm_cos_lr(lr, total_steps, warmup_steps, warmup_lr_start, count):
            """Cosine learning rate with warm up."""
            if count < warmup_steps:
                lr = (lr - warmup_lr_start) * count / float(warmup_steps) + warmup_lr_start
            else:
                lr *= 0.5 * (1.0 + math.cos(math.pi * (count - warmup_steps) / (total_steps - warmup_steps)))
            return lr

        def warm_cos_w_end_lr(lr, total_steps, warmup_steps, warmup_lr_start, end_lr, count):
            """Cosine learning rate with warm up with end lr."""
            if count < warmup_steps:
                lr = (lr - warmup_lr_start) * count / float(warmup_steps) + warmup_lr_start
            else:
                q = 0.5 * (1.0 + math.cos(math.pi * (count - warmup_steps) / (total_steps - warmup_steps)))
                lr = lr * q + end_lr * (1 - q)
            return lr

        warmup_steps = self.warmup_steps
        if warmup_steps == 0:
            return super(_WarmCosineScheduler, self)._get_lr_func()
        warmup_lr_start = self.warmup_lr_start
        end_lr = self.end_lr
        if end_lr > 0:
            lr_func = partial(warm_cos_w_end_lr, self.lr, self.total_steps, warmup_steps, warmup_lr_start, end_lr)
        else:
            lr_func = partial(warm_cos_lr, self.lr, self.total_steps, warmup_steps, warmup_lr_start)
        return lr_func


@LRSCHEDULERS.register(name="warmcos")
class WarmCosineLRScheduler(LRScheduler):
    def __init__(self, optimizer, lr, total_steps, warmup_steps=0, warmup_lr_start=1e-6, end_lr=0.0):
        scheduler = _WarmCosineScheduler(
            lr,
            total_steps,
            warmup_steps=warmup_steps,
            warmup_lr_start=warmup_lr_start,
            end_lr=end_lr,
        )
        super(WarmCosineLRScheduler, self).__init__(optimizer, scheduler)


class _StepScheduler(_Scheduler):
    def __init__(self, lr, total_steps, milestones, gamma=0.1):
        self.milestones = milestones
        self.gamma = gamma
        super(_StepScheduler, self).__init__(lr, total_steps)

    def _get_lr_func(self):
        def multistep_lr(lr, milestones, gamma, count):
            """MultiStep learning rate"""
            for milestone in milestones:
                lr *= gamma if count >= milestone else 1.0
            return lr

        gamma = self.gamma
        lr_func = partial(multistep_lr, self.lr, self.milestones, gamma)
        return lr_func
    
class _WarmStepScheduler(_Scheduler):
    def __init__(self, lr, total_steps, milestones, gamma=0.1, warmup_steps=0.0, warmup_lr_start=1e-6):
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_steps = warmup_steps
        self.warmup_lr_start = warmup_lr_start
        super(_WarmStepScheduler, self).__init__(lr, total_steps)

    def _get_lr_func(self):
        def multistep_lr(lr, milestones, gamma, warmup_steps, warmup_lr_start, count):
            """MultiStep learning rate"""
            if count < warmup_steps:
                lr = (lr - warmup_lr_start) * count / float(warmup_steps) + warmup_lr_start
            else:
                for milestone in milestones:
                    lr *= gamma if count >= milestone else 1.0
            return lr

        gamma = self.gamma
        lr_func = partial(multistep_lr, self.lr, self.milestones, gamma, self.warmup_steps, self.warmup_lr_start)
        return lr_func

@LRSCHEDULERS.register(name="warmmultistep")
class WarmStepLRScheduler(LRScheduler):
    def __init__(self, optimizer, lr, total_steps, milestones, gamma=0.1, warmup_steps=0.0, warmup_lr_start=1e-6):
        scheduler = _WarmStepScheduler(lr, total_steps, milestones, gamma, warmup_steps, warmup_lr_start)
        super(WarmStepLRScheduler, self).__init__(optimizer, scheduler)


@LRSCHEDULERS.register(name="multistep")
class StepLRScheduler(LRScheduler):
    def __init__(self, optimizer, lr, total_steps, milestones, gamma=0.1):
        scheduler = _StepScheduler(lr, total_steps, milestones, gamma)
        super(StepLRScheduler, self).__init__(optimizer, scheduler)


class _ConstantScheduler(_Scheduler):
    def __init__(self, lr, total_steps):
        super(_ConstantScheduler, self).__init__(lr, total_steps)

    def _get_lr_func(self):
        return lambda _: self.lr


@LRSCHEDULERS.register(name="constant")
class ConstantLRScheduler(LRScheduler):
    def __init__(self, optimizer, lr, total_steps):
        scheduler = _ConstantScheduler(lr, total_steps)
        super(ConstantLRScheduler, self).__init__(optimizer, scheduler)
