import os
import numpy as np
import cv2
import shutil
from tqdm import tqdm
import pandas as pd
import glob

from scipy.ndimage import _ni_support
from scipy.ndimage.morphology import distance_transform_edt, binary_erosion, generate_binary_structure
from hce_metric_main import compute_hce


def dict_to_excel(metric_dict, excel_path):
    df = pd.DataFrame.from_dict(metric_dict, orient='index')
    df.to_excel(excel_path)


def __surface_distances(result, reference, voxelspacing=None, connectivity=1):
    """
    The distances between the surface voxel of binary objects in result and their
    nearest partner surface voxel of a binary object in reference.
    """
    result = np.atleast_1d(result.astype(np.bool))
    reference = np.atleast_1d(reference.astype(np.bool))
    if voxelspacing is not None:
        voxelspacing = _ni_support._normalize_sequence(voxelspacing, result.ndim)
        voxelspacing = np.asarray(voxelspacing, dtype=np.float64)
        if not voxelspacing.flags.contiguous:
            voxelspacing = voxelspacing.copy()

    # binary structure
    footprint = generate_binary_structure(result.ndim, connectivity)

    # test for emptiness
    if 0 == np.count_nonzero(result):
        raise RuntimeError('The first supplied array does not contain any binary object.')
    if 0 == np.count_nonzero(reference):
        raise RuntimeError('The second supplied array does not contain any binary object.')

        # extract only 1-pixel border line of objects
    result_border = result ^ binary_erosion(result, structure=footprint, iterations=1)
    reference_border = reference ^ binary_erosion(reference, structure=footprint, iterations=1)

    # compute average surface distance
    # Note: scipys distance transform is calculated only inside the borders of the
    #       foreground objects, therefore the input has to be reversed
    dt = distance_transform_edt(~reference_border, sampling=voxelspacing)
    sds = dt[result_border]

    return sds


def precision(result, reference):
    """
    Precison.

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
    precision : float
        The precision between two binary datasets, here mostly binary objects in images,
        which is defined as the fraction of retrieved instances that are relevant. The
        precision is not symmetric.

    See also
    --------
    :func:`recall`

    Notes
    -----
    Not symmetric. The inverse of the precision is :func:`recall`.
    High precision means that an algorithm returned substantially more relevant results than irrelevant.

    References
    ----------
    .. [1] http://en.wikipedia.org/wiki/Precision_and_recall
    .. [2] http://en.wikipedia.org/wiki/Confusion_matrix#Table_of_confusion
    """
    result = np.atleast_1d(result.astype(np.bool))
    reference = np.atleast_1d(reference.astype(np.bool))

    tp = np.count_nonzero(result & reference)
    fp = np.count_nonzero(result & ~reference)

    try:
        precision = tp / float(tp + fp)
    except ZeroDivisionError:
        precision = 0.0

    return precision


def recall(result, reference):
    """
    Recall.

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
    recall : float
        The recall between two binary datasets, here mostly binary objects in images,
        which is defined as the fraction of relevant instances that are retrieved. The
        recall is not symmetric.

    See also
    --------
    :func:`precision`

    Notes
    -----
    Not symmetric. The inverse of the recall is :func:`precision`.
    High recall means that an algorithm returned most of the relevant results.

    References
    ----------
    .. [1] http://en.wikipedia.org/wiki/Precision_and_recall
    .. [2] http://en.wikipedia.org/wiki/Confusion_matrix#Table_of_confusion
    """
    result = np.atleast_1d(result.astype(np.bool))
    reference = np.atleast_1d(reference.astype(np.bool))

    tp = np.count_nonzero(result & reference)
    fn = np.count_nonzero(~result & reference)

    try:
        recall = tp / float(tp + fn)
    except ZeroDivisionError:
        recall = 0.0

    return recall


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


def hd(result, reference, voxelspacing=None, connectivity=1):
    try:
        hd1 = __surface_distances(result, reference, voxelspacing, connectivity).max()
        hd2 = __surface_distances(reference, result, voxelspacing, connectivity).max()
    except:
        hd = 0
        return hd

    hd = max(hd1, hd2)
    return hd


def jc(result, reference):
    result = np.atleast_1d(result.astype(np.bool_))
    reference = np.atleast_1d(reference.astype(np.bool_))

    intersection = np.count_nonzero(result & reference)
    union = np.count_nonzero(result | reference)

    jc = float(intersection) / float(union)

    return jc


def asd(result, reference, voxelspacing=None, connectivity=1):
    try:
        sds = __surface_distances(result, reference, voxelspacing, connectivity)
    except:
        asd = 0
        return asd
    asd = sds.mean()
    return asd


def assd(result, reference, voxelspacing=None, connectivity=1):
    assd = np.mean(
        (asd(result, reference, voxelspacing, connectivity), asd(reference, result, voxelspacing, connectivity)))
    return assd


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


def conformity(Dice):
    if Dice > 0.01:
        Con = (3 * Dice - 2) / Dice
    else:
        Con = 0.0
    return Con


if __name__ == '__main__':
    pre_root = '/data/liulian/Med_Seg/save_preds/unet_tem/20230217-221311_qulvent_24cat/test4000/image_pred'
    test_source = '/data/liulian/Med_Seg/dataset/test'
    root = '/data/liulian/Med_Seg/save_preds/unet_tem/20230217-221311_qulvent_24cat/test4000/pred_save'
 
    if '.lst' in test_source or '.txt' in test_source:
        with open(test_source, 'r') as f:
            img_lst = [x.strip() for x in f.readlines() if os.path.exists(x.strip())]
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
    p = int(len(img_lst))
    metric_dict = dict()
    for idx in tqdm(range(0, p)):
        i = i+1
        img_path = img_lst[idx]
        img_path = os.path.join(test_source, img_path)
        mask_path = img_path.replace(".png", "_mask.png")
        infer_path = os.path.join(pre_root, img_path.split('/')[-1])

        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, 0)
        h_img ,w_img = img.shape[:2]
        infer = cv2.imread(infer_path, 0)

        # Compute f1
        f1 = F1_score(infer, mask)

        # Compute mae
        mae = MAE(infer, mask)

        # Compute dice
        dice = dc(infer, mask)

        # Compute conformity
        con = conformity(dice)
        
        # Compute hce
        hce = compute_hce(infer, mask)

        # Compute hausdorff distance
        hausdorff_dt = hd(infer, mask)

        # Compute jaccard coefficient
        jaccard_coef = jc(infer, mask)

        # Compute assd
        asd_coef = assd(infer, mask)

        # Compute rvd
        rvd = RVD(infer, mask)

        dice_list.append(dice)
        hd_list.append(hausdorff_dt)
        jc_list.append(jaccard_coef)
        asd_list.append(asd_coef)
        rvd_list.append(rvd)
        f1_list.append(f1)
        mae_list.append(mae)
        con_list.append(con)
        hce_list.append(hce)

        if i % 1000 ==0:
            print("Current calculation up to the {} image".format(i))
        
        # Save the form
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
    # Calculate the average value of the metrics
    with open(txt_save_path,'a+') as f:
        avg_con = np.sum(con_list) / len(img_lst)
        f.write("mean con: %f" % (avg_con * 100))
        f.write('\n')
        avg_dice = np.sum(dice_list) / len(img_lst)
        f.write("mean dice: %f" % (avg_dice * 100))
        f.write('\n')
        avg_jc = np.sum(jc_list) / len(img_lst)
        f.write("mean jc: %f" % (avg_jc * 100))
        f.write('\n')
        avg_f1 = np.sum(f1_list) / len(img_lst)
        f.write("mean f1: %f" % (avg_f1 * 100))
        f.write('\n')

        avg_hce = np.sum(hce_list) / len(img_lst)
        f.write("mean hce: %f" % avg_hce)
        f.write('\n')
        avg_mae = np.sum(mae_list) / len(img_lst)
        f.write("mean mae: %f" % avg_mae)
        f.write('\n')
        avg_hd = np.sum(hd_list) / len(img_lst)
        f.write("mean hd: %f" % avg_hd)
        f.write('\n')
        avg_asd = np.sum(asd_list) / len(img_lst)
        f.write("mean asd: %f" % avg_asd)
        f.write('\n')
        avg_rvd = np.sum(rvd_list) / len(img_lst)
        f.write("mean rvd: %f" % avg_rvd)
        f.write('\n')

