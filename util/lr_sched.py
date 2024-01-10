# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import math

class WarmupCosineAnnealingWarmRestarts():
    def __init__(self, optimizer, args, T_0=20, T_mult=2):
        self.T_0 = T_0
        self.T_i = T_0
        self.T_mult = T_mult
        self.eta_min = args.min_lr
        for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
        self.base_lrs = [group['initial_lr'] for group in optimizer.param_groups]

    def step(self, optimizer, epoch, args):
        if epoch < args.warmup_epochs:
            lr = args.lr * epoch / args.warmup_epochs
            for param_group in optimizer.param_groups:
                if "lr_scale" in param_group:
                    param_group["lr"] = lr * param_group["lr_scale"]
                else:
                    param_group["lr"] = lr
        else:
            delta_epoch = epoch - args.warmup_epochs
            if delta_epoch >= self.T_0:
                n = int(math.log((delta_epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                self.T_cur = delta_epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = delta_epoch

            values = [self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * self.T_cur / self.T_i)) / 2
                for base_lr in self.base_lrs]

            for _, data in enumerate(zip(optimizer.param_groups, values)):
                param_group, lr = data
                if "lr_scale" in param_group:
                    param_group["lr"] = lr * param_group["lr_scale"]
                else:
                    param_group["lr"] = lr
        

def adjust_learning_rate(optimizer, epoch, args, second_start = None):
    total_epochs = args.epochs
    if second_start is not None:
        if epoch < second_start:
            total_epochs = second_start
        else:
            total_epochs = total_epochs - second_start
            epoch = epoch - second_start
            
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs 
    else:
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (total_epochs - args.warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr
