import h5py
import numpy as np
import contextlib

import palettable
import torch
import torch.optim as optim
import open3d as o3d
import matplotlib.pyplot as plt
import numpy
import os
import plotly.express as px
import random
from sklearn.manifold import TSNE
from bhtsne import tsne


def vis_cloud(a,b):
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


def visualize_with_label(cloud, window_name="open3d"):
    # assert cloud.shape[0] == label.shape[0]

    cloud = cloud.reshape((-1, 3))
    #labels = np.asarray(label)
    #max_label = labels.max()
    #colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    #colors = np.squeeze(colors,axis=1)
    pt = o3d.geometry.PointCloud()
    pt.points = o3d.utility.Vector3dVector(cloud)
    pt.paint_uniform_color([0.2, 0.3, 0.9])


    vis = o3d.visualization.Visualizer()
    vis.create_window(width=800, height=800)  # 创建窗口
    render_option: o3d.visualization.RenderOption = vis.get_render_option()  # 设置点云渲染参数
    render_option.background_color = np.array([255, 255, 255])  # 设置背景色（这里为黑色）
    render_option.point_size = 5  # 设置渲染点的大小
    vis.add_geometry(pt)  # 添加点云
    vis.run()



def ISS(cloud):
    # assert cloud.shape[0] == label.shape[0]

    pt = o3d.geometry.PointCloud()
    pt.points = o3d.utility.Vector3dVector(cloud)
    pt.paint_uniform_color([0, 1, 0])
    #pt.paint_uniform_color([0.2, 0.3, 0.9])
    keypoints = o3d.geometry.keypoint.compute_iss_keypoints(pt)
    keypoints.paint_uniform_color([1, 0, 0])
    o3d.visualization.draw([pt,keypoints], point_size=5)




def load_h5(h5_filename, normalize=False, include_label=False):
    f = h5py.File(h5_filename, 'r')
    data = f['data'][:]  # (n, 2048, 3)
    if normalize:
        # nmean = numpy.mean(data, axis=1, keepdims=True)
        # nstd = numpy.std(data, axis=1, keepdims=True)
        # nstd = numpy.mean(nstd, axis=-1, keepdims=True)
        dmin = data.min(axis=1, keepdims=True).min(axis=-1, keepdims=True)
        dmax = data.max(axis=1, keepdims=True).max(axis=-1, keepdims=True)
        data = (data - dmin) / (dmax - dmin)
        # data = (data - nmean) / nstd
        data = 2.0 * (data - 0.5)
    if include_label:
        label = f['label'][:]
        return data, label
    return data


def walk_files(path):
    for r, ds, fs in os.walk(path):
        for f in fs:
            yield os.path.join(r, f)


def all_h5(parent, normalize=False, include_label=False,
           subclasses=tuple(range(40)), sample=None):
    lazy = map(lambda x: load_h5(x, normalize, include_label),
               walk_files(parent))
    if include_label:
        xy = tuple(lazy)
        x = [x for x, y in xy]
        y = [y for x, y in xy]
        #print(x,y)
        x = numpy.concatenate(x)
        y = numpy.concatenate(y)
        xf = []
        yf = []
        for xp, yp in zip(x, y):
            if yp[0] in subclasses:
                if sample is None:
                    xf.append(xp)
                else:
                    xf.append(random.choices(xp, k=sample))
                yf.append(numpy.eye(len(subclasses))[subclasses.index(yp[0])])
        return numpy.array(xf), numpy.array(yf)
    return numpy.concatenate(tuple(lazy))





images = np.load("1+airplane_dm.npy")

# 可视化前4张图像
fig, axes = plt.subplots(4, 2, figsize=(10, 10))  # 2x2的子图布局
axes = axes.ravel()  # 展平axes数组方便遍历

for i in range(8):
    axes[i].imshow(images[i])  # 显示第i张图像
    axes[i].set_title(f"Image {i+1}")
    axes[i].axis("off")  # 关闭坐标轴

plt.tight_layout()
plt.show()


datas,labels = all_h5("./point/train/h5", True, True, subclasses=(27,),sample=None)
data = torch.from_numpy(datas)
print(data.shape)
#visualize_with_label(data[0].reshape([-1,3]))


color = ['#729ECE', '#FF9E4A', '#67BF5C', '#ED665D', '#AD8BC9', '#A8786E', '#ED97CA', '#A2A2A2', '#CDCC5D', '#1F77B4', '#AEC7E8', '#FF7F0E', '#FFBB78', '#2CA02C', '#98DF8A', '#D62728', '#FF9896', '#9467BD', '#C5B0D5', '#8C564B', '#C49C94', '#E377C2', '#F7B6D2', '#7F7F7F', '#C7C7C7', '#BCBD22', '#DBDB8D', '#17BECF', '#9EDAE5']
cnames = palettable.tableau.Tableau_20.hex_colors

def plot_embedding_2D(data, label):

 x_min, x_max = np.min(data, 0), np.max(data, 0)
 data = (data - x_min) / (x_max - x_min)
 fig = plt.figure()
 for i in range(data.shape[0]):
     plt.plot(data[i, 0], data[i, 1], marker='o', markersize=4,color = color[label[i]])
 plt.xticks([])
 plt.yticks([])
 return fig

predicted = np.load('./chair/0819chair_rand.npz')
predicted_kpcd = predicted['kpcd']
predicted_nfact = predicted['nfact']
predicted_lable = np.load('./chair/chairand_lable.npy')
lab  = predicted_lable.reshape([-1,])
kp = predicted_kpcd.reshape([-1,3])

print(kp.shape)
print(lab.shape)

#X_tsne=TSNE(n_components=2).fit_transform(kp)
#X_tsne = tsne(kp,dimensions=2)
#fig1 = plot_embedding_2D(X_tsne,lab)	# 将二维数据用plt绘制出来
#fig1.show()

# plt.figure()
# plt.scatter(X_tsne[:, 0], X_tsne[:, 1],color='y')
# plt.show()
'''
ax = plt.subplot(projection='3d')
for i in range(predicted_kpcd.shape[0]):
    keyp = predicted_kpcd[i]
    #print(keyp.shape)
    for j in range(keyp.shape[0]):
        #print(predicted_lable[i,j])
        #print(keyp[j,0], keyp[j,1], keyp[j,2])
        ax.scatter(keyp[j,0], keyp[j,1], keyp[j,2], s=1.0,c=color[predicted_lable[i,j]])

#
ax.set_xlabel('X', fontweight ='bold')
ax.set_ylabel('Y', fontweight ='bold')
ax.set_zlabel('Z', fontweight ='bold')

ax.set_xlim(-1,1)
ax.set_ylim(-1,1)
ax.set_zlim(-1,1)
#ax.view_init(15, 240)
ax.xaxis.set_major_locator(FixedLocator(np.arange(-1, 1, 0.5)))
ax.xaxis.set_minor_locator(FixedLocator(np.arange(-1, 1, 0.25)))
ax.yaxis.set_major_locator(FixedLocator(np.arange(-1, 1, 0.5)))
ax.yaxis.set_minor_locator(FixedLocator(np.arange(-1, 1, 0.25)))
ax.zaxis.set_major_locator(FixedLocator(np.arange(-1, 1, 0.5)))
ax.zaxis.set_minor_locator(FixedLocator(np.arange(-1, 1, 0.25)))
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
#plt.colorbar(ax, shrink = 0.6, aspect = 5)
#plt.savefig('merger.png')
#plt.show()
# plt.rcParams['grid.color'] = "white"
# fig = plt.figure()
# ax = fig.add_subplot(111,projection = '3d')
# ax.scatter(X_embedded[:,0], X_embedded[:,1], X_embedded[:,2], c=y)
# ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
# ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
# ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

#ax.axis('off')
#plt.show()
'''

######################POINTCONV######################
# model = PointConvDensityClsSsg(num_classes=10)
# x = torch.tensor(datas)
# yy = x[0:12,:,:]
# x = x.permute(0, 2, 1)
# y= x[0:12,:,:]
# output= model(y,y)
# output11 = output[11].permute(1,0)
# output11 = output11.unsqueeze(0)
# vis_cloud(output11,yy[11])
######################POINTCONV######################


####################  ISS  ######################
# x = torch.tensor(datas)
# yy = x[0:12,:,:]
# ISS(yy[11])
####################  ISS  ######################



#airpplane 第12个
#vis_cloud(output11,yy[11])
#visualize_with_label(datas[12],labels)
#18:table
#13 car
#11 camera
#12 cap
#17 laptop
#38 cup
#6 jiuping
#7 Vessel?

