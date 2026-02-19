
import torch
import merger.merger_net as merger_net
import merger.merger_net2 as merger_net1
from merger.model_weightchamfer import Pointnet2StructurePointNet as Net
import json
import tqdm
import numpy as np
import argparse
import open3d as o3d

arg_parser = argparse.ArgumentParser(description="Predictor for Skeleton Merger on KeypointNet dataset. Outputs a npz file with two arrays: kpcd - (N, k, 3) xyz coordinates of keypoints detected; nfact - (N, 2) normalization factor, or max and min coordinate values in a point cloud.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
arg_parser.add_argument('-a', '--annotation-json', type=str, default='../annotations/guitar.json',
                        help='Annotation JSON file path from KeypointNet dataset.')
arg_parser.add_argument('-i', '--pcd-path', type=str, default='../pcds',
                        help='Point cloud file folder path from KeypointNet dataset.')
arg_parser.add_argument('-m', '--checkpoint-path', '--model-path', type=str, default='guitar_old.pt',
                        help='Model checkpoint file path to load.')
arg_parser.add_argument('-d', '--device', type=str, default='cuda',
                        help='Pytorch device for predicting.')
arg_parser.add_argument('-k', '--n-keypoint', type=int, default=10,
                        help='Requested number of keypoints to detect.')
arg_parser.add_argument('-b', '--batch', type=int, default=8,
                        help='Batch size.')
arg_parser.add_argument('--max-points', type=int, default=2048,
                        help='Indicates maximum points in each input point cloud.')
ns = arg_parser.parse_args()

#net = Net(num_structure_points=10, input_channels=0,multi_distribution_num=3,offset=False).to(ns.device)
net = merger_net.Net(2048,10).to(ns.device)
net.load_state_dict(torch.load(ns.checkpoint_path, map_location=torch.device(ns.device))['model_state_dict'])
net.eval()

def rotate_point_cloud_by_angle(batch_data, rotation_angle):
    """ Rotate the point cloud along up direction with certain angle.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        # rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, :, 0:3]
        rotated_data[k, :, 0:3] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data

def compute_iss_repeatability(true_warped_keypoints, warped_keypoints, distance_thresh=0.1):
    N1 = true_warped_keypoints.shape[0]
    N2 = warped_keypoints.shape[0]
    true_warped_keypoints = np.expand_dims(true_warped_keypoints, 1)
    warped_keypoints = np.expand_dims(warped_keypoints, 0)
    # shapes are broadcasted to N1 x N2 x 2:
    norm = np.linalg.norm(true_warped_keypoints - warped_keypoints, ord=None, axis=2)
    count1 = 0
    count2 = 0
    if N2 != 0:
        min1 = np.min(norm, axis=1)
        correct1 = (min1 <= distance_thresh)
        count1 = np.sum(correct1)
    if N1 != 0:
        min2 = np.min(norm, axis=0)
        correct2 = (min2 <= distance_thresh)
        count2 = np.sum(correct2)
    if N1 + N2 > 0:
        repeatability = (count1 + count2) / (N1 + N2)
    else:
        repeatability = -1
    return repeatability

def compute_repeatability(true_warped_keypoints, warped_keypoints, distance_thresh=0.1):
    bs = true_warped_keypoints.shape[0]
    kp_num = true_warped_keypoints.shape[1]
    dist = np.linalg.norm(true_warped_keypoints - warped_keypoints, ord=None, axis=2)
    count = np.sum(dist <= distance_thresh)
    repeatability = count/(bs * kp_num)
    return repeatability

def iss_keypoint_repeatability(point_cloud):
    bs = point_cloud.shape[0]
    repeatability = []
    noise = np.random.normal(scale=0.1, size=(point_cloud).shape)
    point2 = point_cloud + noise
    for i in range(bs):
      pt = o3d.geometry.PointCloud()
      pt.points = o3d.utility.Vector3dVector(point_cloud[i])

      pt2 = o3d.geometry.PointCloud()
      pt2.points = o3d.utility.Vector3dVector(point2[i])

      keypoints = o3d.geometry.keypoint.compute_iss_keypoints(pt)
      keypoints_np = np.asarray(keypoints.points)

      keyp2 = o3d.geometry.keypoint.compute_iss_keypoints(pt2)
      keyp2_np = np.asarray(keyp2.points)

      error = compute_iss_repeatability(keypoints_np,keyp2_np)
      repeatability.append(error)
    rep = sum(repeatability) / len(repeatability)
    return rep


def compute_rre(true_warped_keypoints, warped_keypoints):
    translation_errors = np.linalg.norm(true_warped_keypoints - warped_keypoints, ord=None, axis=2)
    #rmse_translation = np.sqrt(np.mean(translation_errors ** 2))
    error = np.mean(translation_errors, axis=1)
    batch_error = np.sum(error)
    return batch_error

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
    pc = np.array(data[:, :3], dtype=np.float)
    return pc


kpn_ds = json.load(open(ns.annotation_json))
batchsize = []
out_rea = []
out_nfact = []
for i in tqdm.tqdm(range(0, len(kpn_ds), ns.batch), unit_scale=ns.batch):
    Q = []
    for j in range(ns.batch):
        if i + j >= len(kpn_ds):
            continue
        entry = kpn_ds[i + j]
        cid = entry['class_id']
        mid = entry['model_id']
        pc = naive_read_pcd(r'{}/{}/{}.pcd'.format(ns.pcd_path, cid, mid))
        pcmax = pc.max()
        pcmin = pc.min()
        pcn = (pc - pcmin) / (pcmax - pcmin)
        pcn = 2.0 * (pcn - 0.5)
        Q.append(pcn)
        out_nfact.append([pcmax, pcmin])
    if len(Q) == 1:
        Q.append(Q[-1])
        out_nfact.append(out_nfact[-1])
    with torch.no_grad():
        #noise = np.random.normal(size=(np.array(Q).shape))*0.1
        #pc = torch.Tensor(np.array(Q) + noise)
        pc = torch.Tensor(rotate_point_cloud_by_angle(np.array(Q),np.pi/2))
        #iss_kp_error = iss_keypoint_repeatability(np.array(Q))
        #print(iss_kp_error)
        recon1, key_points1, kpa1,null_activation1 = net(pc.to(ns.device))
        recon, key_points, kpa,null_activation = net(torch.Tensor(np.array(Q)).to(ns.device))

        #key_points1, fps_points1,kpa1, null_activation1 = net(pc.to(ns.device))
        #key_points, fps_points, kpa, null_activation = net(torch.Tensor(np.array(Q)).to(ns.device))
        error = compute_rre(np.array(key_points1.cpu().numpy()),np.array(key_points.cpu().numpy()))
        #print(error)
        out_rea.append(error)


average = sum(out_rea) / len(kpn_ds)

print("列表的平均值为:", average)
#
# np.savez(ns.prediction_output, kpcd=out_kpcd, nfact=out_nfact)
