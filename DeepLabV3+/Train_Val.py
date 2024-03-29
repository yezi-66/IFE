import os
import torch
import torch.nn.functional as F
import numpy as np
import logging
from apex import amp
import torch.backends.cudnn as cudnn
from datetime import datetime
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter

from modeling.deeplab import *
from utils.dataset import get_loader
from utils.utils import adjust_lr, dice_coef, structure_loss, plot_image, create_exp_dir


def train(train_loader, model, optimizer, epoch, save_path, writer):
    """
    train function
    """
    model.train()
    loss_all = 0
    epoch_step = 0
    dices = 0.0  #  dice
    num = 0
    total_step = len(train_loader)
    try:
        for i, (images, gts) in enumerate(train_loader, start=1):
            images = images.cuda()
            gts = gts.cuda()

            preds = model(images)
            loss = structure_loss(preds, gts)

            # 增加dice计算
            num += images.shape[0]
            preds_ = preds[-1].squeeze().sigmoid().data.cpu().numpy()
            preds_ = (preds_ - preds_.min()) / (preds_.max() - preds_.min() + 1e-8)
            preds_ = (preds_ >= 0.5)
            gts_ = gts.squeeze().cpu().data.numpy() # 为计算dice先转成numpy
            dice = dice_coef(preds_, gts_)
            dices += (dice * images.shape[0]) # 计算一轮的总dice

            optimizer.zero_grad()
            with amp.scale_loss(loss, optimizer) as scale_loss:
                scale_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # ​clip_grad_norm​​解决梯度爆炸问题
            optimizer.step()

            epoch_step += 1
            loss_all += loss.item()

            if i % 20 == 0 or i == total_step or i == 1:
                print('{}|| Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Total_loss: {:.4f}'.
                      format(datetime.now(), epoch, opt.epoch, i, total_step, loss.data))
                logging.info(
                    '[Train Info]:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Total_loss: {:.4f}'.format(
                        epoch, opt.epoch, i, total_step, loss.data))
                # TensorboardX-Loss
                writer.add_scalars('Loss_Statistics',
                                   {'Loss_total': loss.data},
                                   global_step=i)

        epoch_avg_dice = dices / num
        epoch_avg_loss = loss_all / epoch_step # 与求每一批的loss总和相加再除以图像数量的计算结果一样
    except KeyboardInterrupt:
        print('Keyboard Interrupt: save model and exit.')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(model.state_dict(), os.path.join(save_path, 'Net_epoch_{}.pth'.format(epoch + 1)))
        print('Save checkpoints successfully!')
        raise

    return epoch_avg_loss, epoch_avg_dice


def val(test_loader, model):
    """
    validation function
    """
    model.eval()
    losses = 0.0  #  损失
    dices = 0.0  #  dice
    num = 0
    with torch.no_grad():
        for i, (image, gt) in enumerate(test_loader):
            image, gt = image.cuda(), gt.cuda()

            pred = model(image)
            loss_total = structure_loss(pred, gt)

            # 增加loss画图
            loss = loss_total * image.shape[0]
            losses += loss.item()  #  求loss总和
            # 增加dice计算
            num += image.shape[0]
            pred_ = pred.squeeze().sigmoid().data.cpu().numpy()
            pred_ = (pred_ - pred_.min()) / (pred_.max() - pred_.min() + 1e-8)
            pred_ = (pred_ >= 0.5)
            gt_ = gt.squeeze().cpu().data.numpy()
            dice = dice_coef(pred_, gt_)
            dices += (dice * image.shape[0]) # 计算一轮的总dice
        
        epoch_avg_dice = dices / num
        epoch_avg_loss = losses / num
        
        return epoch_avg_loss, epoch_avg_dice


def main(args):
    # build the model
    model = DeepLab(num_classes=1, ratio_list = args.ratio_list, mode = args.mode,
                        backbone=args.backbone,
                        output_stride=args.out_stride,
                        sync_bn=args.sync_bn,
                        freeze_bn=args.freeze_bn).cuda()

    if args.load is not None:
        model.load_state_dict(torch.load(args.load))
        print('load model from ', args.load)

    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    model, optimizer = amp.initialize(model, optimizer, opt_level='O1')

    save_path = os.path.join(args.save_path, args.save_name)
    print(save_path)
    os.makedirs(os.path.join(save_path, 'weight'), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'logs'), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'summary'), exist_ok=True)

    # load data
    print('Load dataset.......')
    train_loader = get_loader(batchsize = args.batchsize, trainsize = args.trainsize, file=args.train_file, mode='train')
    val_loader = get_loader(batchsize = args.vbatchsize, trainsize = args.trainsize, file=args.val_file, mode='valid')

    # logging
    logging.basicConfig(filename=os.path.join(save_path, 'logs', f'{args.save_name}.log'),
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
    logging.info("Network-Train")
    logging.info("GPU: {}".format(args.gpu_id))
    logging.info("ratio: {}".format(args.ratio_list))
    logging.info("save_name: {}".format(args.save_name))
    logging.info("mode: {}".format(args.mode))
    logging.info('Dataset: train: {}; val: {}'.format(args.train_file, args.val_file))
    logging.info('Config: epoch: {}; lr: {}; batchsize: {}; trainsize: {}; decay_rate: {}; decay_epoch: {}; load: {}; '
                 'save_path: {}'.format(args.epoch, args.lr, args.batchsize, args.trainsize,
                                        args.decay_rate, args.decay_epoch, args.load, save_path))

    writer = SummaryWriter(os.path.join(save_path,'summary'))

    epoch_losses = []
    epoch_dices = []
    epoch_val_losses = []
    epoch_val_dices = []
    best_dice = 0
    best_epoch = 1

    print("Start train......")
    for epoch in range(1, args.epoch+1):
        cur_lr = adjust_lr(optimizer, args.lr, epoch, args.decay_rate, args.decay_epoch)
        writer.add_scalar('learning_rate', cur_lr, global_step=epoch)
        # 训练
        loss_t, dice_t = train(train_loader, model, optimizer, epoch, os.path.join(save_path, 'weight'), writer)
        epoch_losses.append(loss_t)
        epoch_dices.append(dice_t)
        logging.info('[Train Info]: Epoch [{:03d}/{:03d}], Loss_AVG: {:.4f}, Train_Dice: {:.4f}'.format(epoch, args.epoch+1, loss_t, dice_t))
        writer.add_scalar('Loss-epoch', loss_t, global_step=epoch)
        # 验证
        loss_v, dice_v = val(val_loader, model)
        epoch_val_losses.append(loss_v)
        epoch_val_dices.append(dice_v)
        writer.add_scalar('DICE', torch.tensor(dice_v), global_step=epoch)
        if dice_v > best_dice:
            best_dice = dice_v
            best_epoch = epoch
            torch.save(model.state_dict(), f"{save_path}/weight/Net_epoch{epoch}_bestdice{best_dice:.4f}.pth")
            print('Save bestmae state_dict successfully! Best epoch:{}.'.format(epoch))

        print('Epoch: {}, Dice: {}, bestDice: {}, bestEpoch: {}'.format(epoch, dice_v, best_dice, epoch))
        logging.info(
            '[Val Info]:Epoch:{} bestEpoch:{}, bestDice: {}, Val_Dice: {}, Val_Loss: {}'.format(epoch, best_epoch, best_dice, dice_v, loss_v))
        plot_image(os.path.join(save_path,'logs'), epoch_losses, epoch_dices, epoch_val_losses, epoch_val_dices)
        
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', type=str, default='/data/ll/Med_Seg/dataset_full/train', help='train_lst')
    parser.add_argument('--val_file', type=str, default='/data/ll/Med_Seg/dataset_full/val', help='val_lst')
    parser.add_argument('--epoch', type=int, default=60, help='epoch number')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--batchsize', type=int, default=32, help='training batch size')
    parser.add_argument('--vbatchsize', type=int, default=32, help='validing batch size')
    parser.add_argument('--trainsize', type=list, default=[224, 256, 288, 320, 352, 384] , help='training dataset size of resize')
    parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int, default=30, help='every n epochs decay learning rate')
    parser.add_argument('--load', type=str, default=None, help='train from checkpoints') # 继续训练的模型
    parser.add_argument('--gpu_id', type=str, default='1', help='train use gpu')
    parser.add_argument('--mode', type=str, default='curvature', help='optional modes: ori, curvature, and entropy')
    parser.add_argument('--ratio_list', type=list, default=[0.75, 0.75], help='Selection ratio from shallow to deep layers')
    parser.add_argument('--save_path', type=str, default='/data/ll/Med_Seg/deeplab_v3/github/save_path', help='the path to save model, figure and log')
    parser.add_argument('--save_name', type=str, default='dl_75', help='the name of saved dir')
    parser.add_argument('--backbone', type=str, default='resnet',
                        choices=['resnet', 'xception', 'drn', 'mobilenet'],
                        help='backbone name (default: resnet)')
    parser.add_argument('--out-stride', type=int, default=16,
                        help='network output stride (default: 8)')
    parser.add_argument('--use-sbd', action='store_true', default=True,
                        help='whether to use SBD dataset (default: True)')
    parser.add_argument('--sync-bn', type=bool, default=None,
                        help='whether to use sync bn (default: auto)')
    parser.add_argument('--freeze-bn', type=bool, default=False,
                        help='whether to freeze bn parameters (default: False)')
    opt = parser.parse_args()

    # set the device for training
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
    print(opt)
    cudnn.benchmark = True

    main(opt)
