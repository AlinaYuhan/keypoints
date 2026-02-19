import cv2
import torch
import merger.strategy as strategy
import merger.merger_net as merger_net
import merger.merger_net2 as merger_net2
from merger.model_weightchamfer import Pointnet2StructurePointNet as distribution_net
from  merger.data_flower import all_h5,load_h5
import json
import numpy as np
import open3d as o3d
import argparse

arg_parser = argparse.ArgumentParser(description="Predictor for Skeleton Merger on KeypointNet dataset. Outputs a npz file with two arrays: kpcd - (N, k, 3) xyz coordinates of keypoints detected; nfact - (N, 2) normalization factor, or max and min coordinate values in a point cloud.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
arg_parser.add_argument('-m', '--checkpoint-path', '--model-path', type=str, default='airplane_k32.pt',#0819chair_rand 0924chair_k12 0321chair_udak
                        help='Model checkpoint file path to load.')
arg_parser.add_argument('-d', '--device', type=str, default='cuda',
                        help='Pytorch device for predicting.')
arg_parser.add_argument('--max-points', type=int, default=2048,
                        help='Indicates maximum points in each input point cloud.')
ns = arg_parser.parse_args()

#net = distribution_net(num_structure_points=10, input_channels=0,multi_distribution_num=3,offset=False).to(ns.device)
net = merger_net2.Net1(2048,32).to(ns.device)
#net = merger_net.Net(2048,10).to(ns.device)
#net = strategy(2048,10).to(ns.device)

net.load_state_dict(torch.load(ns.checkpoint_path, map_location=torch.device(ns.device))['model_state_dict'])
net.eval()

def visual_clouds(a,b):
    # a:n*3的矩阵
    # b:n*3的矩阵
    pt1=o3d.geometry.PointCloud()
    pt1.points=o3d.utility.Vector3dVector(a)
    pt1.paint_uniform_color([1,0,0])

    pt2 = o3d.geometry.PointCloud()
    pt2.points = o3d.utility.Vector3dVector(b.reshape(-1, 3))
    pt2.paint_uniform_color([0, 1, 0])
    #
    o3d.visualization.draw_geometries([pt1, pt2], window_name='cloud[0] and corr', width=800, height=600)



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
    pc = np.array(data[:, :3], dtype=float)
    colors = np.array(data[:, -1], dtype=int)
    a, indices = np.unique(colors, return_index=True)
    colors = np.stack([(colors >> 16) & 255, (colors >> 8) & 255, colors & 255], -1)
    return pc,colors

def visualize_with_label(cloud, window_name="open3d"):
    # assert cloud.shape[0] == label.shape[0]

    cloud = cloud.reshape((-1, 3))
    pt = o3d.geometry.PointCloud()
    pt.points = o3d.utility.Vector3dVector(cloud)
    #pt.paint_uniform_color([0.5,0.5,0.5])
    i = 0
    spheres = o3d.geometry.TriangleMesh()
    for keypoint in pt.points:
           sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
           sphere.translate(keypoint)
           sphere.paint_uniform_color([0.6,0.6,0.6])#airp:0.53725,0.40784,0.80392  #guita:0,0,0.80392  skateboard:0.18039,0.5451,0.34118
           spheres += sphere
           i=i+1


    vis = o3d.visualization.Visualizer()
    vis.create_window(width=800, height=800)  # 创建窗口
    render_option: o3d.visualization.RenderOption = vis.get_render_option()  # 设置点云渲染参数
    render_option.background_color = np.array([255, 255, 255])  # 设置背景色（这里为黑色）
    render_option.point_size = 6  # 设置渲染点的大小
    vis.add_geometry(spheres)  # 添加点云
    vis.run()



def vis_cloud(a,b):

    a = a.reshape((-1, 3))
    pt1=o3d.geometry.PointCloud()
    pt1.points=o3d.utility.Vector3dVector(a)
    pt1.paint_uniform_color([0.9,0.9,0.9])

    #pt1.colors = o3d.utility.Vector3dVector(colors / 282.)
    pt2 = o3d.geometry.PointCloud()
    pt2.points = o3d.utility.Vector3dVector(b.reshape(-1, 3))
    i = 0
    spheres = o3d.geometry.TriangleMesh()
    for keypoint in pt2.points:
           sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.035)
           sphere.translate(keypoint)
           sphere.paint_uniform_color([0.18039,0.5451,0.34118])#airp:0.53725,0.40784,0.80392  #guita:0,0,0.80392  skateboard:0.18039,0.5451,0.34118
           spheres += sphere
           i=i+1
    o3d.visualization.draw_geometries([pt1, spheres], window_name='cloud[0] and corr', width=800, height=600)

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



json_path = '../annotations/table.json'
pcd_path = '../pcds'
kpn_ds = json.load(open(json_path))
entry = kpn_ds[0]#chair:258 140 101 200 104 138_16 airplane:52_k16 58_k16 83_k16 126_k16 86 guitar:49 28 33 27 guitar: 43 27 25 10 table:0 8
cid = entry['class_id']
mid = entry['model_id']
pc,colors = naive_read_pcd(r'{}/{}/{}.pcd'.format(pcd_path, cid, mid))
gtkp = entry['keypoints']
ground_truths = []
for kp in gtkp:
     a = pc[kp['pcd_info']['point_index']]
     ground_truths.append(a)
b = np.array(ground_truths).reshape([-1,3])
print(b.shape)
gt = torch.tensor(b)#.unsqueeze(0).float()


# aa = b[0:2]
# aaa = b[0:9]
# aaaa = b[10:12]
# result = np.concatenate((aaa, aaaa), axis=0)
# b = result

pcmax = pc.max()
pcmin = pc.min()
pcn= (pc - pcmin) / (pcmax - pcmin)
pcn = 2.0 * (pcn - 0.5)
pcn = torch.tensor(pcn)
pcn = pcn.unsqueeze(0).cuda().float()

b = (b - pcmin) / (pcmax - pcmin)
b = 2.0 * (b - 0.5)




datas,labels = all_h5("./point/train/h5", True, True, subclasses=(13,),sample=None)
#xxx = torch.tensor(datas[18:19,:,:]).cuda()#chair:5-7 10:12 18:20 rec:6:8 airplane:41-43 55-57 66-68 guitar:67:69 60:62 31:33 50:52 42:44 skateb: 33:34 15:16 26:27 7:8 table:14:15 18:19 30:31

data,labe = load_h5("./point/train/h5/train0.h5",normalize=True, include_label=True)
print(datas.shape)
# pcd = o3d.io.read_point_cloud("../../../mitsuba2/PointFlowRenderer-master/skate1_pc.pcd")
# pc = torch.tensor(np.asarray(pcd.points)).unsqueeze(0).cuda()
# xxx = pc

# noise = np.random.normal(size=(xxx.shape))*0.05
# xxx = xxx + torch.Tensor(noise).cuda()

#yyy = rotate_point_cloud_by_angle(pcn.cpu().numpy(),30)
#visual_clouds(pcn.cpu().numpy().reshape([-1,3]),yyy.reshape([-1,3]))
# noise = np.random.normal(scale=0.1, size=(pcn.shape))
# pcn = pcn + torch.Tensor(noise).cuda()
xxx = pcn

points = np.load("cat_vertices.npy")
points = torch.tensor(points).unsqueeze(0).cuda()
print(points.shape)
with torch.no_grad():
        recon, key_points, emb, strength = net(xxx.float().to(ns.device))#,gt.to(ns.device))



#pcn = xxx.detach().cpu().numpy().reshape([-1,3])
reconstruct = recon[0].detach().cpu().numpy().reshape([-1,3])
key_points = key_points[0].detach().cpu().numpy().reshape([-1,3])
xxx=xxx[0].detach().cpu().numpy().reshape([-1,3])

#key_points = np.delete(key_points, [3,5,15,13,2,10], axis=0)
#vis_cloud(xxx,key_points)
#visualize_with_label(datas[0])




point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(xxx)
# keypoints = o3d.geometry.keypoint.compute_iss_keypoints(point_cloud)
# keypoints.paint_uniform_color([1, 0, 0])
# o3d.visualization.draw([point_cloud, keypoints], point_size=5)
#o3d.io.write_point_cloud("../../../mitsuba2/PointFlowRenderer-master/table2_udak.pcd", point_cloud)





