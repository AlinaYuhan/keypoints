import json
import numpy as np
import collections
import argparse
import torch
import  math

arg_parser = argparse.ArgumentParser(description="Evaluation for detected keypoints.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
arg_parser.add_argument('-a', '--annotation-json', type=str, default='../annotations/chair.json',
                        help='Annotation JSON file path from KeypointNet dataset.')
arg_parser.add_argument('-i', '--pcd-path', type=str, default='../pcds',
                        help='Point cloud file folder path from KeypointNet dataset.')
arg_parser.add_argument('-p', '--prediction', type=str, default='chair_nolc.npz',#./airplane/0818airplane_k32.npz 0321chair_udak
                        help='Prediction file from predictor output.')
arg_parser.add_argument('--op-align-fwd', action='store_false',
                        help='Computes forward alignment score.')
arg_parser.add_argument('--op-align-bwd', action='store_false',
                        help='Computes backward alignment score.')
arg_parser.add_argument('--op-miou', action='store_false',
                        help='Draws mIoU curve with matplotlib.')


def naive_read_pcd(path):
    lines = open(path, 'r').readlines()
    idx = -1
    for i, line in enumerate(lines):
        if line.startswith('DATA ascii'):
            idx = i + 1
            break
    lines = lines[idx:]
    lines = [line.rstrip().split(' ') for line in lines]
    data = np.asarray(lines)
    pc = np.array(data[:, :3], dtype=np.float64)
    return pc


def fwd_alignment_scores():
    preds = []
    for entry, kpcd, nfact in zip(kpn_ds, predicted['kpcd'], predicted['nfact']):
        dmax = nfact[0]
        dmin = nfact[1]
        ground_truths = []
        gtkp = entry['keypoints']
        for kp in gtkp:
            nkp = (kp['xyz'] - dmin) / (dmax - dmin)
            nkp = 2.0 * (nkp - 0.5)
            ground_truths.append(nkp)
        ground_truths = np.array(ground_truths)  # k2 x 3
        kpcd_e = np.expand_dims(kpcd, 1)  # k1 x 1 x 3
        gt_e = np.expand_dims(ground_truths, 0)  # 1 x k2 x 3
        dist = np.sum(np.square(kpcd_e - gt_e), -1)  # k1 x k2
        argminfwd = np.argmin(dist, -1)  # k1 np.argmin()求最小值对应的索引
        preds.append([gtkp[argm]['semantic_id'] for argm in argminfwd])
    preds = np.array(preds, dtype=np.int32)  # n x k1
    acc = []
    for pa in preds:
        for pb in preds:
            acc.append(np.mean(pa == pb))
    return preds,np.mean(acc)


def bwd_alignment_scores():
    preds = collections.defaultdict(list)
    for entry, kpcd, nfact in zip(kpn_ds, predicted['kpcd'], predicted['nfact']):
        dmax = nfact[0]
        dmin = nfact[1]
        ground_truths = []
        gtkp = entry['keypoints']
        for kp in gtkp:
            nkp = (kp['xyz'] - dmin) / (dmax - dmin)
            nkp = 2.0 * (nkp - 0.5)
            ground_truths.append(nkp)
        ground_truths = np.array(ground_truths)  # k2 x 3
        kpcd_e = np.expand_dims(kpcd, 1)  # k1 x 1 x 3
        gt_e = np.expand_dims(ground_truths, 0)  # 1 x k2 x 3
        dist = np.sum(np.square(kpcd_e - gt_e), -1)  # k1 x k2
        argminbwd = np.argmin(dist, -2)  # k2
        for i in range(len(gtkp)):
            sem = gtkp[i]['semantic_id']
            preds[sem].append(argminbwd[i])
    q = []
    for semarr in preds.values():
        semarr = np.array(semarr, dtype=np.int16)
        q.append(np.mean(np.expand_dims(semarr, -1) == np.expand_dims(semarr, 0)))
    return np.mean(q)


def mIoU(thresholds):
    kps = []
    gts = []

    for entry, kpcd, nfact in zip(kpn_ds, predicted['kpcd'], predicted['nfact']):

        cid = entry['class_id']
        mid = entry['model_id']
        pc = naive_read_pcd(r'{}/{}/{}.pcd'.format(ns.pcd_path, cid, mid))
        dmax = nfact[0]
        dmin = nfact[1]
        ground_truths = []
        gtkp = entry['keypoints']
        for kp in gtkp:
            ground_truths.append(pc[kp['pcd_info']['point_index']])
        gts.append(ground_truths)
        npc = (pc - dmin) / (dmax - dmin)
        npc = 2.0 * (npc - 0.5)
        kpcd_e = np.expand_dims(kpcd, 1)  # k1 x 1 x 3
        npc_e = np.expand_dims(npc, 0)  # 1 x k2 x 3
        dist = np.sqrt(np.sum(np.square(kpcd_e - npc_e), -1))  # k1 x k2
        argminfwd = np.argmin(dist, -1)  # k1
        kps.append(pc[argminfwd])
    for threshold in thresholds:
        npos = 0
        fp_sum = 0
        fn_sum = 0
        for ground_truths, kpcd in zip(gts, kps):
            kpcd_e = np.expand_dims(kpcd, 1)  # k1 x 1 x 3
            gt_e = np.expand_dims(ground_truths, 0)  # 1 x k2 x 3
            dist = np.sqrt(np.sum(np.square(kpcd_e - gt_e), -1))  # k1 x k2
            npos += len(np.min(dist, -2))
            fp_sum += np.count_nonzero(np.min(dist, -1) > threshold)
            fn_sum += np.count_nonzero(np.min(dist, -2) > threshold)
        yield (npos - fn_sum) / (npos + fp_sum)


def mIoU_curve_plot():
    import matplotlib.pyplot as plotlib
    #plotlib.style.use('seaborn')
    miou_curve = list(mIoU(np.linspace(0., 0.1)))
    print(miou_curve)
    plotlib.plot(np.linspace(0., 0.1), miou_curve,color = 'red')
    plotlib.grid(True, linestyle='--', alpha=0.5)
    #plotlib.legend()
    plotlib.show()

def calc_pck(thresholds):
    kps = []
    gts = []
    for entry, kpcd, nfact in zip(kpn_ds, predicted['kpcd'], predicted['nfact']):
        cid = entry['class_id']
        mid = entry['model_id']
        pc = naive_read_pcd(r'{}/{}/{}.pcd'.format(ns.pcd_path, cid, mid))
        dmax = nfact[0]
        dmin = nfact[1]
        ground_truths = []
        gtkp = entry['keypoints']
        for kp in gtkp:
            ground_truths.append(pc[kp['pcd_info']['point_index']])
        gts.append(ground_truths)
        npc = (pc - dmin) / (dmax - dmin)
        npc = 2.0 * (npc - 0.5)
        kpcd_e = np.expand_dims(kpcd, 1)  # k1 x 1 x 3
        npc_e = np.expand_dims(npc, 0)  # 1 x k2 x 3
        dist = np.sqrt(np.sum(np.square(kpcd_e - npc_e), -1))  # k1 x k2
        argminfwd = np.argmin(dist, -1)  # k1
        kps.append(pc[argminfwd])

    q = []
    for ground_truths, kpcd in zip(gts, kps):
        bb = [];x=[];y=[]
        kpcd_e = np.expand_dims(kpcd, 1)  # k1 x 1 x 3
        gt_e = np.expand_dims(ground_truths, 0)  # 1 x k2 x 3
        dist = np.sqrt(np.sum(np.square(kpcd_e - gt_e), -1))  # k1 x k2
        aa=np.min(dist, -1)
        argminbwd1 = np.argmin(dist, -1)
        i = 0
        for a in aa:
            if a <= thresholds:
                x.append(a)
                y.append(i)
            i = i+1
        for c in y:
            bb.append(argminbwd1[c])
        a, indices = np.unique(bb, return_index=True)
        mea = len(a)/(len(ground_truths))
        q.append(mea)

    return np.mean(q)





def calc_pck1(gt_keypoints, est_keypoints, threshold=0.3):
    """The PCK (Percentage of Correct Keypoints) metric is a common evaluation metric
    used to assess the accuracy of pose estimation algorithms.
    It measures the percentage of keypoints (joints) whose projection error is
    less than a certain threshold (default 0.3).
    The projection error is the Euclidean distance between
    the ground truth keypoint location and the estimated keypoint location in the image plane.

    Args:
        gt_keypoints (ndarray): ground truth keypoints (num_keypoints, 2)
        est_keypoints (ndarray): estimated keypoints
        threshold (float, optional): threshold. Defaults to 0.3.

    Returns:
        float: the PCK@threshold score
    """
    num_keypoints = gt_keypoints.shape[0]
    pck = np.zeros(num_keypoints)
    for i in range(num_keypoints):
        gt_kp = gt_keypoints[i]
        est_kp = est_keypoints[i]
        error = np.linalg.norm(gt_kp - est_kp)
        if error < threshold:
            pck[i] = 1
    return np.mean(pck)



if __name__ == '__main__':
    ns = arg_parser.parse_args()
    with open(ns.annotation_json) as data_file:
        kpn_ds = json.load(data_file)
    predicted = np.load(ns.prediction,allow_pickle=True)
    #if ns.op_align_fwd:
        #lab,score = fwd_alignment_scores()
        #print(lab.shape)
        #np.save("motorcyclerand_lable.npy", lab)
        #print("Forward Alignment:", score)
    #if ns.op_align_bwd:
        #print("Backward Alignment:", bwd_alignment_scores())
    if ns.op_miou:
        mIoU_curve_plot()

    #pck = calc_pck(0.05)
    #print(pck)

