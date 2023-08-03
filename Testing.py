import torch
import torch.nn.functional as F
import tqdm
import os, argparse
import cv2

from utils.dataset import MedDataset

from modeling.deeplab import *
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

parser = argparse.ArgumentParser()
parser.add_argument('--test_file', type=str, default='/data2/sod_data/test_sample_quarter_new.lst', help='test_lst')
parser.add_argument('--testsize', type=int, default=256, help='testing size')
parser.add_argument('--pth_path', type=str, default='/data2/zhouhan/SINet/deeplab_v3/save_path/dl_wc_75_low_entro/weight/Net_epoch59_bestdice0.9346.pth')
parser.add_argument('--mode', type=str, default='entro')
parser.add_argument('--ratio_list', type=list, default=[0.75,0], help='the path to save model, figure and log')
parser.add_argument('--backbone', type=str, default='resnet',
                    choices=['resnet', 'xception', 'drn', 'mobilenet'],
                    help='backbone name (default: resnet)')
parser.add_argument('--out-stride', type=int, default=16,
                    help='network output stride (default: 8)')
parser.add_argument('--use-sbd', action='store_true', default=True,
                    help='whether to use SBD dataset (default: True)')
parser.add_argument('--sync-bn', type=bool, default=None,
                    help='whether to use sync bn (default: auto)')
opt = parser.parse_args()

save_path = os.path.join(opt.pth_path.split('/weight')[0], "image_pred")
print(save_path)

os.makedirs(save_path, exist_ok=True)

model = DeepLab(num_classes=1,ratio_list = opt.ratio_list, mode = opt.mode,
                        backbone=opt.backbone,
                        output_stride=opt.out_stride,
                        sync_bn=opt.sync_bn,
                        freeze_bn=False).cuda()

model.load_state_dict(torch.load(opt.pth_path))
model.cuda()
model.eval()
test_loader = MedDataset(trainsize = opt.testsize, file=opt.test_file, mode='test')
for i, (image, shape, name) in enumerate(tqdm.tqdm(test_loader)):
    H, W = shape
    # s = max(H, W)
    # h_pad_0 = (s - H) // 2
    # w_pad_0 = (s - W) // 2
    # h_pad_1 = s - H - h_pad_0
    # w_pad_1 = s - W - w_pad_0 
    image = image.cuda()
    image = torch.unsqueeze(image, 0).float()

    res = model(image)
    res = F.interpolate(res, (H, W), mode='bilinear', align_corners=True)
    # res = res[0, 0][h_pad_0:(s-h_pad_1), w_pad_0:(s-w_pad_1)]
    res = res.sigmoid().data.cpu().numpy()
    # # normalize
    res = (res - res.min()) / (res.max() - res.min() + 1e-8)
    res = (res >= 0.5)
    cv2.imwrite(os.path.join(save_path, name), res * 255) 
        
