import os
import shutil
import torch
import numpy as np
import torch.nn.functional as F
from thop import profile
from thop import clever_format
import matplotlib.pyplot as plt
from pathlib import Path


def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=30):
    decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = decay*init_lr
        lr=param_group['lr']
    return lr


def create_exp_dir(path, scripts_path_to_save=None):
    if not os.path.exists(path):
        os.mkdir(path)
    print('Experiment dir : {}'.format(path))
    p = Path(scripts_path_to_save)
    py_list = list(p.glob("*.py")) + list(p.glob("**/*.py"))

    for root in py_list: 
        py_path = str(root)
        if "save" in py_path:
            continue
        elif "lib" in py_path:
            save_path = os.path.join(path, "code", "lib")
            
        elif "utils" in py_path:
            save_path = os.path.join(path, "code", "utils")
        else:
            save_path = os.path.join(path, "code")
        os.makedirs(save_path, exist_ok=True)
        dst_file = os.path.join(save_path, py_path.split('/')[-1])
        shutil.copyfile(py_path, dst_file)

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

        # 数据可视化
        # 训练集损失
        plt.figure(figsize=(6, 6))
        plt.plot(losses, lw=1.5)
        plt.title('Train Loss')
        plt.xlabel('Epoch Number')
        plt.ylabel('Loss')
        plt.savefig(f'{path}/train_loss.png')

        # 训练集dice系数
        plt.figure(figsize=(6, 6))
        plt.plot(dices, lw=1.5)
        plt.title('Train Dice')
        plt.xlabel('Epoch Number')
        plt.ylabel('Dice')
        plt.savefig(f'{path}/train_dice.png')

        # 验证集损失
        plt.figure(figsize=(6, 6))
        plt.plot(val_losses, lw=1.5)
        plt.title('Valid Loss')
        plt.xlabel('Epoch Number')
        plt.ylabel('Loss')
        plt.savefig(f'{path}/valid_loss.png')

        # 验证集dice系数
        plt.figure(figsize=(6, 6))
        plt.plot(val_dices, lw=1.5)
        plt.title('Valid Dice')
        plt.xlabel('Epoch Number')
        plt.ylabel('Dice')
        plt.savefig(f'{path}/valid_dice.png')


class AvgMeter(object):
    def __init__(self, num=40):
        self.num = num
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.losses = []

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.losses.append(val)

    def show(self):
        return torch.mean(torch.stack(self.losses[np.maximum(len(self.losses)-self.num, 0):]))


def CalParams(model, input_tensor):
    """
    Usage:
        Calculate Params and FLOPs via [THOP](https://github.com/Lyken17/pytorch-OpCounter)
    Necessarity:
        from thop import profile
        from thop import clever_format
    :param model:
    :param input_tensor:
    :return:
    """
    flops, params = profile(model, inputs=(input_tensor,))
    flops, params = clever_format([flops, params], "%.3f")
    print('[Statistics Information]\nFLOPs: {}\nParams: {}'.format(flops, params))