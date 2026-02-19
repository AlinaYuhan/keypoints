
import torch
import torch.nn as nn
from timm.models.layers import DropPath,trunc_normal_
import torch.nn.functional as F
#from pointnet2_ops import pointnet2_utils
from  merger.anchor_points_test import GeoCrossAttention,Attention,Mlp


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)   #transpose只能对两个维度进行转换
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)

    return idx


def get_graph_feature(x, k=20, idx=None, dim9=False):

    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points).cuda()

    if idx is None:
        if dim9 == False:
            idx = knn(x, k=k)#.cuda()  # (batch_size, num_points, k)
        else:
            idx = knn(x[:, 6:], k=k)

    idx_base = (torch.arange(0, batch_size).view(-1, 1, 1)*num_points).cuda()
    idx = (idx + idx_base).cuda()
    idx = idx.view(-1)
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()#.cuda()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]#.cuda()
    feature = feature.view(batch_size, num_points, k, num_dims)#.cuda()
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)#.cuda()

    edge_fea = (feature-x).permute(0, 3, 1, 2).contiguous()
    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()  # .cuda()
  
    return feature      # (batch_size, 2*num_dims, num_points, k)


def get_graph_feature1(x, k=20, idx=None, dim9=False):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points).cuda()

    if idx is None:
        if dim9 == False:
            idx = knn(x, k=k)  # .cuda()  # (batch_size, num_points, k)
        else:
            idx = knn(x[:, 6:], k=k)

    idx_base = (torch.arange(0, batch_size).view(-1, 1, 1) * num_points).cuda()
    idx = (idx + idx_base).cuda()
    idx = idx.view(-1)
    _, num_dims, _ = x.size()

    x = x.transpose(2,1).contiguous()  # .cuda()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]  # .cuda()
    feature = feature.view(batch_size, num_points, k, num_dims)  # .cuda()
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)  # .cuda()

    edge_fea = (feature - x).permute(0, 3, 1, 2).contiguous()
    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()  # .cuda()

    return feature  # (batch_size, 2*num_dims, num_points, k)


class DGCNN_cls(nn.Module):
    def __init__(self, output_channels,drop_path=0.):
        super(DGCNN_cls, self).__init__()
        #self.args = args
        self.k = 20
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(output_channels)#default=1024
        self.bn6 = nn.BatchNorm1d(1024)
        #self.bn7 = nn.BatchNorm1d(384)


        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2)).cuda()

        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.conv5 = nn.Sequential(nn.Conv1d(512, output_channels, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(nn.Conv1d(512, 1024, kernel_size=1, bias=False),
                                   self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))
        # self.conv7 = nn.Sequential(nn.Conv1d(1024, 384, kernel_size=1, bias=False),
        #                            self.bn7,
        #                            nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(1024*2, output_channels, bias=False)

        self.dp1 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(512, 256)

        self.dp2 = nn.Dropout(p=0.5)
        self.linear3 = nn.Linear(128, output_channels)

        self.norm1 = nn.LayerNorm(1024)

        self.mlp = Mlp(in_features=1024, hidden_features=1024, act_layer=nn.GELU, drop=0.)

        self.self_attn = Attention(dim=1024, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
                               proj_drop=0.)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.merge_map = nn.Linear(1024 * 2, 1024)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            nn.init.xavier_normal_(m.weight.data, gain=1)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight.data, 1)
            nn.init.constant_(m.bias.data, 0)


    def forward(self, x):

        N = x.shape[2]
        batch_size = x.size(0)

        x = get_graph_feature(x, k=self.k)# (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        x = self.conv1(x)  # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]# (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x1, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv2(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)


        x = get_graph_feature(x2, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 128, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)


        x = get_graph_feature(x3, k=self.k)     # (batch_size, 128, num_points) -> (batch_size, 128*2, num_points, k)
        x = self.conv4(x)                       # (batch_size, 128*2, num_points, k) -> (batch_size, 256, num_points, k)
        x4 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 256, num_points, k) -> (batch_size, 256, num_points)


        x = torch.cat((x1, x2, x3, x4), dim=1)  # (batch_size, 64+64+128+256, num_points)
        gx = self.conv6(x)

        norm_x = self.norm1(gx.permute(0, 2, 1))
        x_1 = self.self_attn(norm_x)
        xx = gx + self.drop_path(x_1).permute(0, 2, 1)
        xx = xx.permute(0, 2, 1) + self.drop_path(self.mlp(self.norm1(xx.permute(0, 2, 1))))

        x = self.conv5(x)  # (batch_size, 64+64+128+256, num_points) -> (batch_size, emb_dims, num_points)

        globalf = torch.max(xx, dim=1, keepdim=False)[0]
        global_f = F.adaptive_max_pool1d(gx, 1).view(batch_size, -1)  # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)

        glo_f = torch.cat([globalf,globalf],dim=-1)
        x_1 = self.merge_map(glo_f)

        x_soft = F.log_softmax(x, dim=1)
        x_soft = x_soft.permute(0, 2, 1)

        #g_f = F.adaptive_max_pool1d(fff, 1).view(batch_size,-1)

        #x = self.linear1(x)
        #x = F.leaky_relu(self.bn6(x), negative_slope=0.2) # (batch_size, emb_dims*2) -> (batch_size, 512)
        # x = self.dp1(x)
        # x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2) # (batch_size, 512) -> (batch_size, 256)
        # x = self.dp2(x)
        # x = self.linear3(x)                                             # (batch_size, 256) -> (batch_size, output_channels)
        
        return x_soft,x_1


# class DGCNN_group(nn.Module):
#     def __init__(self, output_channels=40):
#         super(DGCNN_group, self).__init__()
#
#         self.bn1 = nn.BatchNorm2d(64)
#         self.bn2 = nn.BatchNorm2d(64)
#         self.bn3 = nn.BatchNorm2d(128)
#         self.bn4 = nn.BatchNorm2d(256)
#         self.bn5 = nn.BatchNorm1d(1024)
#
#
#         self.input_trans = nn.Conv1d(3, 8, 1).cuda()
#
#         self.layer1 = nn.Sequential(nn.Conv2d(16, 32, kernel_size=1, bias=False),
#                                    nn.GroupNorm(4, 32),
#                                    nn.LeakyReLU(negative_slope=0.2)
#                                    ).cuda()
#
#         self.layer2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
#                                    nn.GroupNorm(4, 64),
#                                    nn.LeakyReLU(negative_slope=0.2)
#                                    ).cuda()
#
#         self.layer3 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1, bias=False),
#                                    nn.GroupNorm(4, 64),
#                                    nn.LeakyReLU(negative_slope=0.2)
#                                    ).cuda()
#
#         self.layer4 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=1, bias=False),
#                                    nn.GroupNorm(4, 128),
#                                    nn.LeakyReLU(negative_slope=0.2)
#                                    ).cuda()
#         self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
#                                    self.bn1,
#                                    nn.LeakyReLU(negative_slope=0.2)).cuda()
#         self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
#                                    self.bn2,
#                                    nn.LeakyReLU(negative_slope=0.2)).cuda()
#         self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
#                                    self.bn3,
#                                    nn.LeakyReLU(negative_slope=0.2)).cuda()
#         self.conv4 = nn.Sequential(nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
#                                    self.bn4,
#                                    nn.LeakyReLU(negative_slope=0.2)).cuda()
#         self.conv5 = nn.Sequential(nn.Conv1d(480, 1024, kernel_size=1, bias=False),
#                                    self.bn5,
#                                    nn.LeakyReLU(negative_slope=0.2)).cuda()
#         self.linear1 = nn.Linear(1024 * 2, 1024, bias=False).cuda()
#
#
#     @staticmethod
#     def fps_downsample(coor, x, num_group):
#         xyz = coor.transpose(1, 2).contiguous()  # b, n, 3
#         fps_idx = pointnet2_utils.furthest_point_sample(xyz, num_group)
#
#         combined_x = torch.cat([coor, x], dim=1)
#
#         new_combined_x = (
#             pointnet2_utils.gather_operation(
#                 combined_x, fps_idx
#             )
#         )
#
#         new_coor = new_combined_x[:, :3]
#         new_x = new_combined_x[:, 3:]
#
#         return new_coor, new_x
#
#     def forward(self, x):
#         batch_size = x.size(0)
#
#         f = self.input_trans(x)
#         f = get_graph_feature(f, k=16)
#         f = self.layer1(f)
#         f = f.max(dim=-1, keepdim=False)[0]
#         # x1 = f
#
#         coor_q, f_q = self.fps_downsample(x, f, 512)
#         f = get_graph_feature(f_q, k=16)
#         f = self.layer2(f)
#         f = f.max(dim=-1, keepdim=False)[0]
#         coor = coor_q
#         # x = get_graph_feature(x1, k=16)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
#         # x = self.conv2(x)
#         # x2 = x.max(dim=-1, keepdim=False)[0]
#
#         f = get_graph_feature(f, k=16)
#         f = self.layer3(f)
#         f = f.max(dim=-1, keepdim=False)[0]
#         # x = get_graph_feature(x2, k=16)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
#         # x = self.conv3(x)
#         # x3 = x.max(dim=-1, keepdim=False)[0]
#
#         coor_q, f_q = self.fps_downsample(coor, f, 128)
#         f = get_graph_feature(f_q, k=16)
#         f = self.layer4(f)
#         f = f.max(dim=-1, keepdim=False)[0]
#         coor = coor_q
#         # x = get_graph_feature(x3, k=16)  # (batch_size, 128, num_points) -> (batch_size, 128*2, num_points, k)
#         # x = self.conv4(x)  # (batch_size, 128*2, num_points, k) -> (batch_size, 256, num_points, k)
#         # x4 = x.max(dim=-1, keepdim=False)[0]
#
#         # x = torch.cat((x1, x2, x3, x4), dim=1)
#         # x = self.conv5(x)  # (batch_size, 64+64+128+256, num_points) -> (batch_size, emb_dims, num_points)
#         # x1 = F.adaptive_max_pool1d(x, 1).view(batch_size,-1)  # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
#         # x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size,-1)  # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
#         #
#         # x = torch.cat((x1, x2), 1)
#         # x = F.leaky_relu(self.bn5(self.linear1(x)), negative_slope=0.2).cuda()
#
#         return coor, f


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net  = DGCNN_cls(10).to(device)
    x = torch.randn(3,3,2048).cuda()
    y,gf= net(x)
    print(y.shape,gf.shape)

