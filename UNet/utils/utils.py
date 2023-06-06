import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt


def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=30):
    decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = decay*init_lr
        lr=param_group['lr']
    return lr


def dice_coef(result, reference):
    result = np.atleast_1d(result.astype(np.bool_))
    reference = np.atleast_1d(reference.astype(np.bool_))

    intersection = np.count_nonzero(result & reference)

    size_i1 = np.count_nonzero(result)
    size_i2 = np.count_nonzero(reference)

    try:
        dc = 2. * intersection / float(size_i1 + size_i2)
    except ZeroDivisionError:
        dc = 0.0

    return dc


def structure_loss(pred, mask):
    """
    loss function (ref: F3Net-AAAI-2020)
    """
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='mean')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def plot_image(path, epoch_losses, epoch_dices, epoch_val_losses, epoch_val_dices):        
    losses = np.array(epoch_losses)
    dices = np.array(epoch_dices)
    val_losses = np.array(epoch_val_losses)
    val_dices = np.array(epoch_val_dices)

    plt.figure(figsize=(6, 6))
    plt.plot(losses, lw=1.5)
    plt.title('Train Loss')
    plt.xlabel('Epoch Number')
    plt.ylabel('Loss')
    plt.savefig(f'{path}/train_loss.png')

    plt.figure(figsize=(6, 6))
    plt.plot(dices, lw=1.5)
    plt.title('Train Dice')
    plt.xlabel('Epoch Number')
    plt.ylabel('Dice')
    plt.savefig(f'{path}/train_dice.png')

    plt.figure(figsize=(6, 6))
    plt.plot(val_losses, lw=1.5)
    plt.title('Valid Loss')
    plt.xlabel('Epoch Number')
    plt.ylabel('Loss')
    plt.savefig(f'{path}/valid_loss.png')

    plt.figure(figsize=(6, 6))
    plt.plot(val_dices, lw=1.5)
    plt.title('Valid Dice')
    plt.xlabel('Epoch Number')
    plt.ylabel('Dice')
    plt.savefig(f'{path}/valid_dice.png')
