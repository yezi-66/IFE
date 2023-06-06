import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
import cv2
import tqdm

from utils.dataset import MedDataset
from lib.Network import Network

parser = argparse.ArgumentParser()
parser.add_argument('--test_file', type=str, default='./test', help='test set path or record file')
parser.add_argument('--testsize', type=int, default=256, help='testing size')
parser.add_argument('--pth_path', type=str, default='')
parser.add_argument('--mode', type=str, default='ori', help='optional modes: ori, curvature, and entropy')
parser.add_argument('--ratio_list', type=list, default=[0.5, 0.5], help='Selection ratio from shallow to deep layers')
opt = parser.parse_args()

save_path = os.path.join(opt.pth_path.split('/weight')[0], "image_pred")
print(save_path)
os.makedirs(save_path, exist_ok=True)

model = Network(mode=opt.mode, ratio=opt.ratio_list)
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
    cv2.imwrite(os.path.join(save_path, name), res * 255) 
        
