
import numpy as np
import open3d as o3d
import argparse
import torch
import merger.merger_net2 as merger_net2

arg_parser = argparse.ArgumentParser(description="Predictor for Skeleton Merger on KeypointNet dataset. Outputs a npz file with two arrays: kpcd - (N, k, 3) xyz coordinates of keypoints detected; nfact - (N, 2) normalization factor, or max and min coordinate values in a point cloud.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
arg_parser.add_argument('-m', '--checkpoint-path', '--model-path', type=str, default='410chair.pt',#0819chair_rand 0924chair_k12 0321chair_udak
                        help='Model checkpoint file path to load.')
arg_parser.add_argument('-d', '--device', type=str, default='cuda',
                        help='Pytorch device for predicting.')
arg_parser.add_argument('--max-points', type=int, default=2048,
                        help='Indicates maximum points in each input point cloud.')
ns = arg_parser.parse_args()

net = merger_net2.Net1(2048,10).to(ns.device)
net.load_state_dict(torch.load(ns.checkpoint_path, map_location=torch.device(ns.device))['model_state_dict'])
net.eval()
def visualize_with_label(cloud, colors,window_name="open3d"):
    # assert cloud.shape[0] == label.shape[0]
    color2 = np.zeros([5, 3])
    c = np.zeros(cloud.shape)
    color2[0] = [0, 0, 0]
    color2[1] = [0, 0, 1]
    color2[2] = [1, 0, 0]
    color2[3] = [0, 1, 0]
    color2[4] = [1, 0, 1]
    a = 0
    for i in colors:
        col = color2[i]
        c[a] =col
        a = a+1
    cloud = cloud.reshape((-1, 3))
    pt = o3d.geometry.PointCloud()
    pt.points = o3d.utility.Vector3dVector(cloud)
    pt.colors = o3d.utility.Vector3dVector(c)


    vis = o3d.visualization.Visualizer()
    vis.create_window(width=800, height=800)  # 创建窗口
    render_option: o3d.visualization.RenderOption = vis.get_render_option()  # 设置点云渲染参数
    render_option.background_color = np.array([255, 255, 255])  # 设置背景色（这里为黑色）
    render_option.point_size = 5  # 设置渲染点的大小
    vis.add_geometry(pt)  # 添加点云
    vis.run()

def vis_cloud(pt,colors,b):

    color1 = np.zeros([16, 3])
    color1[0] = [0, 0, 1]
    color1[1] = [0, 0.74902, 1]
    color1[2] = [1, 0, 1]
    color1[3] = [0, 0, 1]
    color1[4] = [1, 0, 0]
    color1[5] = [1, 0.54902, 0]
    color1[6] = [1, 1, 0]
    color1[7] = [0, 1, 0]
    color1[8] = [0, 1, 1]
    color1[9] = [0.41569, 0.35294, 0.80392]
    color1[10] = [0.30588, 0.93333, 0.58039]
    color1[11] = [0.51373, 0.43529, 1]
    color1[12] = [0.56471, 0.93333, 0.56471]
    color1[13] = [1, 0.20392, 0.70196]
    color1[14] = [1, 0.27059, 0]
    color1[15] = [0.5451, 0.29804, 0.22353]
    color2 = np.zeros([5, 3])
    c = np.zeros(pt.shape)
    color2[0] = [0, 0, 0]
    color2[1] = [0.49804,1,0.83137]
    color2[2] = [0,0.96078,1]
    color2[3] = [1,0.8549,0.72549]
    color2[4] = [0.81176,0.81176,0.81176]
    a = 0
    for i in colors:
        col = color2[i]
        c[a] =col
        a = a+1
    cloud = pt.reshape((-1, 3))
    pt1 = o3d.geometry.PointCloud()
    pt1.points = o3d.utility.Vector3dVector(cloud)
    pt1.colors = o3d.utility.Vector3dVector(c)


    pt2 = o3d.geometry.PointCloud()
    pt2.points = o3d.utility.Vector3dVector(b.reshape(-1, 3))
    i = 0
    spheres = o3d.geometry.TriangleMesh()
    for keypoint in pt2.points:
           sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.03)
           sphere.translate(keypoint)
           #sphere.paint_uniform_color(color1[i])
           sphere.paint_uniform_color([1,0,0])
           spheres += sphere
           i=i+1

    #pt2.paint_uniform_color([0, 1, 0])
    #
    o3d.visualization.draw_geometries([pt1, spheres], window_name='cloud[0] and corr', width=800, height=600)




#
# data_path = '../shapenetcore/04379243/points'
# label_path = '../shapenetcore/04379243/points_label'
# # 采样点
# NUM_SAMPLE_POINTS = 2048
# # 存储点云与label
# point_clouds = []
# point_clouds_labels = []
# file_list = os.listdir(data_path)
i = 14
x_set = np.load("shapenet_guitar.npy")
label = np.load("shapenet_guitar_label.npy")
x1 = torch.tensor(x_set[i]).unsqueeze(0)
label1=label[i]
print(label1)

with torch.no_grad():
    recon, key_points, emb, null_activation = net(x1.float().to(ns.device))
pcn = x1.detach().cpu().numpy().reshape([-1,3])
reconstruct = recon[0].detach().cpu().numpy().reshape([-1,3])
key_points = key_points[0].detach().cpu().numpy().reshape([-1,3])
vis_cloud(pcn,label1,key_points)
#
# for file_name in tqdm.tqdm(file_list):
# #     # 获取label和data的地址
#      label_name = file_name.replace('.pts', '.seg')
#      point_cloud_file_path = os.path.join(data_path, file_name)
# #     label_file_path = os.path.join(label_path, label_name)
# #     # 读取label和data
#      point_cloud = np.loadtxt(point_cloud_file_path)
#      print(point_cloud.shape)
# #     label = np.loadtxt(label_file_path).astype('int')
# #     # 如果本身的点少于需要采样的点，则直接去除
#      if len(point_cloud) < NUM_SAMPLE_POINTS:
#          continue
# #     # 采样
#      num_points = len(point_cloud)
# #     # 确定随机采样的index
#      sampled_indices = random.sample(list(range(num_points)), NUM_SAMPLE_POINTS)
# #     # 点云采样
#      sampled_point_cloud = np.array([point_cloud[i] for i in sampled_indices])
# #     # label采样
# #     sampled_label_cloud = np.array([label[i] for i in sampled_indices])
# #     # 正则化
#      norm_point_cloud = sampled_point_cloud - np.mean(sampled_point_cloud, axis=0)
# #     norm_point_cloud /= np.max(np.linalg.norm(norm_point_cloud, axis=1))
# #
# #     # 存储
#      point_clouds.append(norm_point_cloud)
# #     point_clouds_labels.append(sampled_label_cloud)
# #
# #
# point_clouds = np.array(point_clouds)
# # point_clouds_labels = np.array(point_clouds_labels)
# # print(point_clouds.shape)
# np.save("shapenet_table.npy",point_clouds)
# np.save("../../SkeletonMerger-main/SkeletonMerger-main/shapenet_car_label.npy",point_clouds_labels)
