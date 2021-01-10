import numpy as np
import torch
import shutil
import os
from .remote_op import remote_scp
from multiprocessing import Process


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, model, path, big_server=False, mode='min', compare=None, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.big_server = big_server
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = 0.0
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

        self.model = model
        self.path = path
        self.mode = mode
        if compare is None:
            if mode == 'min':
                self.compare = lambda a, b: a < b
            elif mode == 'max':
                self.compare = lambda a, b: a > b
            else:
                assert 0
        else:
            self.compare = compare

    def __call__(self, score):
        if self.best_score == 0 or self.compare(score, self.best_score + self.delta):
            if self.verbose:
                print("{:.6} improved to {:.6}".format(self.best_score, score))
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        self.save_checkpoint(score)
        return self.early_stop

    def save_checkpoint(self, score):
        path = "{}_{:6}.ckpt".format(self.path[:-5], int(score * 1000000))
        torch.save(self.model.state_dict(), path)
