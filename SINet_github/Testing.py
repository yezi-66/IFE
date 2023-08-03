import torch
import torch.nn.functional as F
import tqdm
import os, argparse
import cv2

from utils.dataset import MedDataset
from lib.Network_Res2Net_GRA_NCD import Network
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

parser = argparse.ArgumentParser()
parser.add_argument('--test_file', type=str, default='/data2/sod_data/test_sample_half.lst', help='test_lst')
parser.add_argument('--testsize', type=int, default=256, help='testing size')
parser.add_argument('--pth_path', type=str, default='/data2/zhouhan/SINet/sinet_ori_resize/save_path/entro_75_1_half/weight/Net_epoch59_bestdice0.9342.pth')
parser.add_argument('--mode', type=str, default='entro')
parser.add_argument('--ratio_list', type=list, default=[0.75,0.75,1], help='the path to save model, figure and log')
opt = parser.parse_args()

save_path = os.path.join(opt.pth_path.split('/weight')[0], "image_pred_new")
print(save_path)

os.makedirs(save_path, exist_ok=True)

model = Network(mode = opt.mode, ratio_list = opt.ratio_list)
model.load_state_dict(torch.load(opt.pth_path))
model.cuda()
model.eval()
test_loader = MedDataset(trainsize = opt.testsize, file=opt.test_file, mode='test')
for i, (image, shape, name) in enumerate(tqdm.tqdm(test_loader)):
    # if "Kidney" not in name:
    #     continue
    H, W = shape
    s = max(H, W)
    h_pad_0 = (s - H) // 2
    w_pad_0 = (s - W) // 2
    h_pad_1 = s - H - h_pad_0
    w_pad_1 = s - W - w_pad_0 
    image = image.cuda()
    image = torch.unsqueeze(image, 0).float()

    res4, res3, res2, res1 = model(image)
    res = res1
    res = F.interpolate(res, (s, s), mode='bilinear', align_corners=True)
    res = res[0, 0][h_pad_0:(s-h_pad_1), w_pad_0:(s-w_pad_1)]
    res = res.sigmoid().data.cpu().numpy()
    # # normalize
    res = (res - res.min()) / (res.max() - res.min() + 1e-8)
    res = (res >= 0.5)
    cv2.imwrite(os.path.join(save_path, name), res * 255) 
        
