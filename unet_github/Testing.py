import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
import cv2
import tqdm

from utils.dataset import MedDataset
from lib.Network import Network

parser = argparse.ArgumentParser()
parser.add_argument('--test_file', type=str, default='/data/liulian/Med_Seg/dataset/dice_choose/hard', help='test list')
parser.add_argument('--testsize', type=int, default=256, help='testing size')
parser.add_argument('--checkpoint_path', type=str, default='/data/liulian/Med_Seg/train_output/unet_tem/20230217-221311_qulvent_24cat/weight/Net_epoch68_bestdice0.8961.pth')
parser.add_argument('--pred_save_dir', type=str, default='/data/liulian/Med_Seg/save_preds/unet_tem/20230217-221311_qulvent_24cat/hard/image_pred')
opt = parser.parse_args()
os.makedirs(opt.pred_save_dir, exist_ok=True)

model = Network()
model.load_state_dict(torch.load(opt.pth_path))
model.cuda()
model.eval()
test_loader = MedDataset(trainsize = opt.testsize, file=opt.test_file, mode='test')
for i, (image, shape, name) in enumerate(tqdm.tqdm(test_loader)):
    H, W = shape
    s = max(H, W)
    h_pad_0 = (s - H) // 2
    w_pad_0 = (s - W) // 2
    h_pad_1 = s - H - h_pad_0
    w_pad_1 = s - W - w_pad_0 
    image = image.cuda()
    image = torch.unsqueeze(image, 0).float()

    res = model(image)
    res = F.interpolate(res, (s, s), mode='bilinear', align_corners=True)
    res = res[0, 0][h_pad_0:(s-h_pad_1), w_pad_0:(s-w_pad_1)]
    res = res.sigmoid().data.cpu().numpy()
    # # normalize
    res = (res - res.min()) / (res.max() - res.min() + 1e-8)
    res = (res >= 0.5)
    cv2.imwrite(os.path.join(opt.pred_save_dir, name), res * 255) 
        
