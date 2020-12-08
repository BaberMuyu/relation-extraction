import torch
import torch.nn as nn


def loss0(logits, y_true, mask):
    m = 0.49
    u = 0.8
    th0 = logits < 0.5 + m
    th1 = logits > 0.5 - m
    th2 = th0 * th1
    th3 = torch.sign((logits - 0.5) * (y_true - 0.5)) < 0
    th = (th2 + th3).float()
    mask = mask.unsqueeze(dim=1)
    mask = mask * th

    logits_loss = nn.functional.binary_cross_entropy(logits, y_true, reduction="none")
    mask0 = mask * y_true
    mask1 = mask * (1 - y_true)
    logits_loss0 = (logits_loss * mask0).sum().div(mask0.sum())
    logits_loss1 = u * (logits_loss * mask1).sum().div(mask1.sum())
    logits_loss = logits_loss0 + logits_loss1

    return logits_loss


def sl_loss(logits, gt, A, a, b):
    ce_loss = nn.functional.binary_cross_entropy(logits, gt, reduction="none")
    rce_loss = gt * (logits - 1) * A + (gt - 1) * logits * A
    # rce_loss = 0
    loss = a * ce_loss + b * rce_loss
    return loss
