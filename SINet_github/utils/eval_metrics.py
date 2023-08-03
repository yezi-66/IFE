import os
import numpy as np
import cv2
import shutil
from tqdm import tqdm
import pandas as pd
import glob
from seg_eva import __surface_distances, recall, precision
from hce_metric_main import compute_hce


def dict_to_excel(metric_dict, excel_path):
    df = pd.DataFrame.from_dict(metric_dict, orient='index')
    df.to_excel(excel_path)

# dice系数
def dc(result, reference):
    r"""
    Dice coefficient

    Computes the Dice coefficient (also known as Sorensen index) between the binary
    objects in two images.

    The metric is defined as

    .. math::

        DC=\frac{2|A\cap B|}{|A|+|B|}

    , where :math:`A` is the first and :math:`B` the second set of samples (here: binary objects).

    Parameters
    ----------
    result : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    reference : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.

    Returns
    -------
    dc : float
        The Dice coefficient between the object(s) in ```result``` and the
        object(s) in ```reference```. It ranges from 0 (no overlap) to 1 (perfect overlap).

    Notes
    -----
    This is a real metric. The binary images can therefore be supplied in any order.
    """
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

# 豪斯多夫距离
def hd(result, reference, voxelspacing=None, connectivity=1):
    try:
        hd1 = __surface_distances(result, reference, voxelspacing, connectivity).max()
        hd2 = __surface_distances(reference, result, voxelspacing, connectivity).max()
    except:
        hd = 0
        return hd

    hd = max(hd1, hd2)
    return hd

# 杰卡德相似系数
def jc(result, reference):
    result = np.atleast_1d(result.astype(np.bool_))
    reference = np.atleast_1d(reference.astype(np.bool_))

    intersection = np.count_nonzero(result & reference)
    union = np.count_nonzero(result | reference)

    jc = float(intersection) / float(union)

    return jc


# 平均表面距离
def asd(result, reference, voxelspacing=None, connectivity=1):
    try:
        sds = __surface_distances(result, reference, voxelspacing, connectivity)
    except:
        asd = 0
        return asd
    asd = sds.mean()
    return asd


# 平均对称表面距离
def assd(result, reference, voxelspacing=None, connectivity=1):
    assd = np.mean(
        (asd(result, reference, voxelspacing, connectivity), asd(reference, result, voxelspacing, connectivity)))
    return assd


# 相对体积差
def RVD(result, reference):
    result = np.atleast_1d(result.astype(np.bool_))
    reference = np.atleast_1d(reference.astype(np.bool_))

    vol1 = np.count_nonzero(result)
    vol2 = np.count_nonzero(reference)

    if 0 == vol2:
        raise RuntimeError('The second supplied array does not contain any binary object.')

    return 100 * np.abs(vol1 / vol2 - 1)


def F1_score(result,reference):
	pre = precision(result, reference)
	sen = recall(result, reference)
	f1_score = (1+0.3)*pre*sen/(0.3*pre+sen + 1e-4)

	return f1_score

def MAE(result,reference):
    mae_sum = np.sum(np.abs(result - reference)) * 1.0 / ((reference.shape[0] * reference.shape[1] * 255.0) + 1e-4)

    return mae_sum

# def conformity(result, reference):
#     result = np.atleast_1d(result.astype(np.bool_))
#     reference = np.atleast_1d(reference.astype(np.bool_))

#     tp = np.count_nonzero(result & reference)

#     fp = np.count_nonzero(result ^ reference)
#     try:
#         con = (1-float(fp)/tp)
#     except ZeroDivisionError:
#         con = 0.0

#     return con

def conformity(Dice): ## 输入输出均为1分制
    if Dice > 0.01:
        Con = (3 * Dice - 2) / Dice
    else:
        Con = 0.0
    return Con

if __name__ == '__main__':
    pre_root = '/data/liulian/Med_Seg/save_preds/sinet/20230222-122341_tem2/test4000/image_pred'
    test_source = '/data/liulian/Med_Seg/dataset/test'
    root = '/data/liulian/Med_Seg/save_preds/sinet/20230222-122341_tem2/test4000/pred_save'
    save_root = os.path.join(root, 'all')
    check_root = os.path.join(root, 'check')
    os.makedirs(save_root, exist_ok=True)
    os.makedirs(check_root, exist_ok=True)
    if '.lst' in test_source:
        with open(test_source, 'r') as f:
            img_lst = [x.strip() for x in f.readlines()]
    else:
        img_lst = glob.glob(f"{test_source}/*")
        img_lst = [i for i in img_lst if 'mask' not in i]

    f1_list = []
    mae_list = []
    con_list = []
    hce_list = []
    dice_list = []
    hd_list = []
    jc_list = []
    asd_list = []
    rvd_list = []
    num_organ_dict = {}
    num = {}
    i = 0
    p = int(len(img_lst)) # 计算所有预测图片的dice
    # p = int(len(masklist) * 0.01) # 计算部分预测图片的dice
    # np.random.shuffle(masklist)
    metric_dict = dict()
    for idx in tqdm(range(0, p)):
        i = i+1
        img_path = img_lst[idx]
        img_path = os.path.join(test_source, img_path)
        mask_path = img_path.replace('Image','Mask').replace(".png", "_mask.png")
        infer_path = os.path.join(pre_root, img_path.split('/')[-1])
        # print(mask_path)
        # 导入label（名字和预测图片名字相等且一一对应）
        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, 0)
        h_img ,w_img = img.shape[:2]
        infer = cv2.imread(infer_path, 0)

        # 计算f1（接近1）
        f1 = F1_score(infer, mask)

        # 计算mae（接近1）
        mae = MAE(infer, mask)

        # 计算dice（接近1）
        dice = dc(infer, mask)

        # 计算conformity（取大）
        con = conformity(dice)
        
        # 计算hce（取小）
        hce = compute_hce(infer, mask)

        # 计算hausdorff distance（取大）
        hausdorff_dt = hd(infer, mask)
        # print('hd = %f' % hausdorff_dt)

        # 计算jaccard coefficient（接近1）
        jaccard_coef = jc(infer, mask)
        # print('jc = %f' % jaccard_coef)

        # 计算平均对称表面距离（取小）
        asd_coef = assd(infer, mask)
        # print('asd = %f' % asd_coef)

        # 计算相对体积差
        rvd = RVD(infer, mask)
        # print('rvd = %f' % rvd)

        dice_list.append(dice)
        hd_list.append(hausdorff_dt)
        jc_list.append(jaccard_coef)
        asd_list.append(asd_coef)
        rvd_list.append(rvd)
        f1_list.append(f1)
        mae_list.append(mae)
        con_list.append(con)
        hce_list.append(hce)
        num_organ = img_path.split('/')[-1].split("_")[0] + '_' +  img_path.split('/')[-1].split("_")[1]

        if not (num_organ in num_organ_dict.keys()):
            num_organ_dict[num_organ] = []
            num_organ_dict[num_organ].extend([dice, hausdorff_dt, jaccard_coef, asd_coef, rvd, f1, mae, con, hce])
            num[num_organ] = 1
        else:
            num_organ_dict[num_organ][0] += dice
            num_organ_dict[num_organ][1] += hausdorff_dt
            num_organ_dict[num_organ][2] += jaccard_coef
            num_organ_dict[num_organ][3] += asd_coef
            num_organ_dict[num_organ][4] += rvd
            num_organ_dict[num_organ][5] += f1
            num_organ_dict[num_organ][6] += mae
            num_organ_dict[num_organ][7] += con
            num_organ_dict[num_organ][8] += hce
            num[num_organ] += 1
        if i % 1000 ==0:
            print("当前计算到第{}张图片".format(i))

        #resize
        h,w = img.shape[:2]
        min_side = min(h,w)
        ratio = 640/min_side
        img = cv2.resize(img,(int(w*ratio), int(h*ratio)))
        mask = cv2.resize(mask,(int(w*ratio), int(h*ratio)))
        infer = cv2.resize(infer,(int(w*ratio), int(h*ratio)))
        img_ori = img.copy()

        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)  #    Find Contour

        #根据图像尺寸来选择画线的粗细
        if len(contours) > 0:  # 增加判断，只有当有轮廓存在时才填充轮廓！
            cv2.drawContours(img, contours, -1, (0, 0, 255), 1)
        contours_p, hierarchy_p = cv2.findContours(infer, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  #    Find Contour
        if len(contours_p) > 0:  # 增加判断，只有当有轮廓存在时才填充轮廓！
            cv2.drawContours(img, contours_p, -1, (0, 255, 0))

        save_img = np.concatenate([img_ori, np.zeros((img_ori.shape[0],10, 3)), img], 1)
        save_img = np.concatenate([save_img, np.zeros((130,save_img.shape[-2], 3))],0)
        h1, w1 = save_img.shape[:2]
        cv2.putText(save_img, f"dice={dice:.3f}", (0, h1-30), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 2)
        cv2.putText(save_img, f"hd={hausdorff_dt:.3f}", (350, h1-30), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 2)
        cv2.putText(save_img, f"con={con:.3f}", (700, h1-30), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 2)
        cv2.putText(save_img, f"hce={hce:.3f}", (0, h1-80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 2)
        cv2.putText(save_img, f"mae={mae:.3f}", (350, h1-80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 2)
        cv2.imwrite(os.path.join(save_root,infer_path.split('/')[-1]),save_img)

        if dice < 0.8:
            shutil.copy(os.path.join(save_root, img_path.split('/')[-1]), os.path.join(check_root, img_path.split('/')[-1])) # 把结果差的导出来看看
        
        ##### 保存预测指标的表格
        name = os.path.basename(img_path)
        metric_dict[name]={}
        metric_dict[name]['dice'] = dice
        metric_dict[name]['con'] = con
        metric_dict[name]['hce'] = hce
        metric_dict[name]['hausdorff'] = hausdorff_dt
        metric_dict[name]['jaccard'] = jaccard_coef
        metric_dict[name]['asd'] = asd_coef
        metric_dict[name]['rvd'] = rvd
        metric_dict[name]['f1'] = f1
        metric_dict[name]['mae'] = mae

    dict_to_excel(metric_dict, os.path.join(root, "metric.xlsx"))
    txt_save_path = os.path.join(root, "result.txt")
    # 计算指标平均值
    with open(txt_save_path,'a+') as f:
        avg_con = np.sum(con_list) / len(img_lst)
        f.write("平均con：%f" % (avg_con * 100))
        f.write('\n')
        avg_dice = np.sum(dice_list) / len(img_lst)
        f.write("平均dice：%f" % (avg_dice * 100))
        f.write('\n')
        avg_jc = np.sum(jc_list) / len(img_lst)
        f.write("平均jc：%f" % (avg_jc * 100))
        f.write('\n')
        avg_f1 = np.sum(f1_list) / len(img_lst)
        f.write("平均f1：%f" % (avg_f1 * 100))
        f.write('\n')
        avg_hce = np.sum(hce_list) / len(img_lst)
        f.write("平均hce：%f" % avg_hce)
        f.write('\n')
        avg_mae = np.sum(mae_list) / len(img_lst)
        f.write("平均mae：%f" % avg_mae)
        f.write('\n')
        avg_hd = np.sum(hd_list) / len(img_lst)
        f.write("平均hd：%f" % avg_hd)
        f.write('\n')
        avg_asd = np.sum(asd_list) / len(img_lst)
        f.write("平均asd：%f" % avg_asd)
        f.write('\n')
        avg_rvd = np.sum(rvd_list) / len(img_lst)
        f.write("平均rvd：%f" % avg_rvd)
        f.write('\n')

        n_lst = ['dice', 'hd', 'jc', 'asd', 'rvd','f1', 'mae', 'con', 'hce']
        for k in num_organ_dict.keys():
            for i in range(len(n_lst)):
                metric = num_organ_dict[k][i] / num[k]
                f.write("数据集{}的平均{}是：{},数据集里的图片有{}张".format(k,n_lst[i], metric,num[k]))
                f.write('\n')
            f.write('\n')
    # f.colse()
