import os
import json
import tqdm
import numpy as np
import h5py


# 打开.h5文件
file = h5py.File('./point/train/h5/train1.h5', 'r')

# 查看.h5文件结构
print("Keys: %s" % file.keys())
file.visit(lambda x: print(x))

# 读取数据
#data = file['path/to/data']
data = file['data'][:]
label1 = file['label'][:]



# 从.npz文件加载点云数据
predicted = np.load('./skateboard/0817skateboard_k8.npz')
# print(predicted.files)
predicted_kpcd = predicted['kpcd']
predicted_nfact = predicted['nfact']
# print(predicted_kpcd)


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
    return pc


json_path = '../annotations/skateboard.json'
pcd_path = '../pcds'
kpn_ds = json.load(open(json_path))
batch_size = 8
# keypoints_color = [1, 0, 0]  # Red color for keypoints
# point_cloud_color = [0, 0, 1]  # Blue color for the original point cloud

output_folder = './skateboard_k8'
os.makedirs(output_folder, exist_ok=True)

for i in tqdm.tqdm(range(0,10, batch_size), unit_scale=batch_size):
     for j in range(batch_size):
         if i + j >= 10:
             continue
         point_cloud_file = os.path.join(output_folder, f"point_cloud_{i + j}.txt")
         keypoints_file = os.path.join(output_folder, f"keypoints_{i+j}.txt")
         entry = kpn_ds[i + j]
         cid = entry['class_id']
         mid = entry['model_id']
         pc = naive_read_pcd(r'{}/{}/{}.pcd'.format(pcd_path, cid, mid))
         pcmax = pc.max()
         pcmin = pc.min()
         pcn = (pc - pcmin) / (pcmax - pcmin)
         pcn = 2.0 * (pcn - 0.5)

         with open(point_cloud_file, 'w') as f:
             for num_points in range(pcn.shape[0]):
                 line = ' '.join(map(str, pcn[num_points]))
                 f.write(line + '\n')

         keypoints = predicted_kpcd[i+j]
         with open(keypoints_file, 'w') as f:
             for num_keypoints in range(keypoints.shape[0]):
                 line = ' '.join(map(str, keypoints[num_keypoints]))
                 f.write(line + '\n')