"""Cosine annealing learning-rate schedule with linear warmup and restarts.

Adapted from
https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup
(see ``cosine_annealing_warmup/scheduler.py``).
"""

from __future__ import annotations

import math
from typing import override

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class CosineAnnealingWarmupRestarts(LRScheduler):
    """Cosine decay between ``min_lr`` and ``max_lr`` with warmup and cyclic restarts.

    Each cycle linearly ramps the learning rate from per-group base values to
    ``max_lr`` over ``warmup_steps``, then cosine-decays back toward ``min_lr``.
    After ``cur_cycle_steps`` optimizer steps the cycle restarts; ``max_lr`` may
    shrink by ``gamma`` each cycle when ``cycle_mult`` lengthens later cycles.

    Args:
        optimizer: Wrapped optimizer (one LR per param group).
        first_cycle_steps: Optimizer steps in the first cosine cycle (including warmup).
        cycle_mult: Multiplier for cycle length after each restart (``1.0`` keeps length fixed).
        max_lr: Peak learning rate at the end of warmup in the first cycle.
        min_lr: Floor learning rate at cosine troughs; also the initial LR after construction.
        warmup_steps: Linear warmup length in optimizer steps
            (must be less than ``first_cycle_steps``).
        gamma: Per-cycle decay factor applied to ``max_lr`` after each restart.
        last_epoch: Index of the last completed step when resuming (``-1`` starts fresh).
    """

    def __init__(
        self,
        optimizer: Optimizer,
        first_cycle_steps: int,
        cycle_mult: float = 1.0,
        max_lr: float = 0.1,
        min_lr: float = 0.001,
        warmup_steps: int = 0,
        gamma: float = 1.0,
        last_epoch: int = -1,
    ) -> None:
        # Remember schedule hyperparameters for cycle bookkeeping.
        self.first_cycle_steps = first_cycle_steps
        self.cycle_mult = cycle_mult
        self.base_max_lr = max_lr
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.gamma = gamma
        # Track the active cycle length and position within it.
        self.cur_cycle_steps = first_cycle_steps
        self.cycle = 0
        self.step_in_cycle = last_epoch
        super().__init__(optimizer, last_epoch)
        # Start every param group at min_lr and record those bases for warmup/cosine.
        self._init_base_lrs()

    def _init_base_lrs(self) -> None:
        """Set each param group to ``min_lr`` and cache the per-group floor values."""
        self.base_lrs: list[float] = []
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.min_lr
            self.base_lrs.append(self.min_lr)

    @override
    def get_lr(self) -> list[float]:
        """Learning rates for the current step within the active cycle."""
        # Before the first step(), defer to the cached base (min) learning rates.
        if self.step_in_cycle == -1:
            return list(self.base_lrs)
        # Linear warmup from each group's base_lr toward max_lr.
        if self.step_in_cycle < self.warmup_steps:
            warmup_progress = self.step_in_cycle / self.warmup_steps
            return [
                (self.max_lr - base_lr) * warmup_progress + base_lr for base_lr in self.base_lrs
            ]
        # Cosine decay from max_lr down toward each group's base_lr after warmup.
        cosine_span = self.cur_cycle_steps - self.warmup_steps
        cosine_progress = (self.step_in_cycle - self.warmup_steps) / cosine_span
        return [
            base_lr + (self.max_lr - base_lr) * (1 + math.cos(math.pi * cosine_progress)) / 2
            for base_lr in self.base_lrs
        ]

    @override
    def step(self, epoch: int | None = None) -> None:
        """Advance the schedule by one optimizer step, or jump to a given step index."""
        if epoch is None:
            # Default path: one optimizer step forward.
            epoch = self.last_epoch + 1
            self.step_in_cycle += 1
            # End of cycle — roll into the next cycle with optional length scaling.
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle -= self.cur_cycle_steps
                self.cur_cycle_steps = (
                    int((self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult)
                    + self.warmup_steps
                )
        elif epoch >= self.first_cycle_steps:
            # Resume or seek: map a global step index into (cycle, step_in_cycle).
            if self.cycle_mult == 1.0:
                self.step_in_cycle = epoch % self.first_cycle_steps
                self.cycle = epoch // self.first_cycle_steps
            else:
                n = int(
                    math.log(
                        epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1,
                        self.cycle_mult,
                    )
                )
                self.cycle = n
                self.step_in_cycle = epoch - int(
                    self.first_cycle_steps * (self.cycle_mult**n - 1) / (self.cycle_mult - 1)
                )
                self.cur_cycle_steps = int(self.first_cycle_steps * self.cycle_mult**n)
        else:
            # Still inside the first cycle when seeking early steps.
            self.cur_cycle_steps = self.first_cycle_steps
            self.step_in_cycle = epoch

        # Peak LR for this cycle may decay with gamma each restart.
        self.max_lr = self.base_max_lr * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        # Push computed LRs into the optimizer param groups.
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr(), strict=True):
            param_group["lr"] = lr
