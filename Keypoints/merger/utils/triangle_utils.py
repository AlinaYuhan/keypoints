import numpy as np
import torch
import copy
from .DistFunc import knn_with_batch
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
from time import *




# batch_index_select
# t (B, n, 3), inds (B, n ,k), return (B, n ,k ,3)
def Batch_index_select(t, dim, inds):
    t = t.unsqueeze(1).expand(t.size(0), t.size(1), t.size(1), t.size(2))  # expand t to (B,n,n,3)
    dummy = inds.unsqueeze(-1).expand(inds.size(0), inds.size(1), inds.size(2), t.size(3))
    out = t.gather(dim, dummy)
    return out


# list_index_select
# t (B, n, 3), ind: list of (?, 3), return: list of (?, 3, 3)
def list_index_select(t, inds):
    bn = t.size()[0]
    triangle = []
    for i in range(bn):
        t_i = t[i]
        triangle_i = t_i[inds[i]]
        triangle.append(triangle_i)
    return triangle


# compute Batch Euclidean Distances of each pair of rows in two tensor a & b
# a (b, m, k); b (b, n, k); output (b, m, n)
def Batch_EuclideanDistances(a, b):
    sq_a = a ** 2
    sum_sq_a = torch.sum(sq_a, dim=2).unsqueeze(2)  # m->[m, 1]
    # print(sum_sq_a.shape)
    sq_b = b ** 2
    sum_sq_b = torch.sum(sq_b, dim=2).unsqueeze(1)  # n->[1, n]
    # print(sum_sq_b.shape)
    bt = b.permute(0, 2, 1)
    return (sum_sq_a + sum_sq_b - 2 * a.bmm(bt) + 1e-14) ** 0.5  # [m, 1] + [1, n] -> [m, n]




def init_graph(shape_xyz, skel_xyz, valid_k=8):
    bn, pn = skel_xyz.size()[0], skel_xyz.size()[1]

    knn_skel = knn_with_batch(skel_xyz, skel_xyz, pn, is_max=False)
    knn_sp2sk = knn_with_batch(shape_xyz, skel_xyz, 3, is_max=False)

    A = torch.zeros((bn, pn, pn)).float().cuda()

    # initialize A with recovery prior: Mark A[i,j]=1 if (i,j) are two skeletal points closest to a surface point
    A[torch.arange(bn)[:, None], knn_sp2sk[:, :, 0], knn_sp2sk[:, :, 1]] = 1
    A[torch.arange(bn)[:, None], knn_sp2sk[:, :, 1], knn_sp2sk[:, :, 0]] = 1


    # initialize A with topology prior
    A[torch.arange(bn)[:, None, None], torch.arange(pn)[None, :, None], knn_skel[:, :, 1:3]] = 1
    A[torch.arange(bn)[:, None, None], knn_skel[:, :, 1:3], torch.arange(pn)[None, :, None]] = 1

    A = torch.triu(A, diagonal=1) + torch.triu(A.transpose(1,2), diagonal=1).transpose(1,2)

    return A

def init_self_graph(skel_xyz, valid_k=8):

    bn, pn = skel_xyz.size()[0], skel_xyz.size()[1]
    knn_skel = knn_with_batch(skel_xyz, skel_xyz, pn, is_max=False)

    A = torch.zeros((bn, pn, pn)).float().cuda()
    
    # initialize A with topology prior
    A[torch.arange(bn)[:, None, None], torch.arange(pn)[None, :, None], knn_skel[:, :, 1:2]] = 1
    A[torch.arange(bn)[:, None, None], knn_skel[:, :, 1:2], torch.arange(pn)[None, :, None]] = 1

    # valid mask: known existing links + knn links
    valid_mask = copy.deepcopy(A)
    valid_mask[torch.arange(bn)[:, None, None], torch.arange(pn)[None, :, None], knn_skel[:, :, 1:valid_k]] = 1
    valid_mask[torch.arange(bn)[:, None, None], knn_skel[:, :, 1:valid_k], torch.arange(pn)[None, :, None]] = 1

    # known mask: known existing links + known absent links, used as the mask to compute binary loss
    known_mask = copy.deepcopy(A)
    known_indice = list(range(valid_k, pn))
    known_mask[torch.arange(bn)[:, None, None], torch.arange(pn)[None, :, None], knn_skel[:, :, known_indice]] = 1
    known_mask[torch.arange(bn)[:, None, None], knn_skel[:, :, known_indice], torch.arange(pn)[None, :, None]] = 1

    A = torch.triu(A, diagonal=1) + torch.triu(A.transpose(1,2), diagonal=1).transpose(1,2)
    valid_mask = torch.triu(valid_mask, diagonal=1) + torch.triu(valid_mask.transpose(1,2), diagonal=1).transpose(1,2)
    known_mask = torch.triu(known_mask, diagonal=1) + torch.triu(known_mask.transpose(1,2), diagonal=1).transpose(1,2)

    return A, valid_mask, known_mask


def batch_triangle_num(A):  # calculate the max triangle number in batch, so other samples need to pad to this size
    A_2 = torch.bmm(A, A)
    A_3 = torch.bmm(A_2, A)
    count = torch.diagonal(A_3, dim1=-2, dim2=-1).sum(1) / 6  # (B), triangle num of each sample
    count = count.long()
    max_num = torch.max(count).item()
    # print("count:", count)
    # print("max_num:", max_num)
    return count, max_num


def batch_triangle_line_num(A):  # calculate the max triangle number in batch, so other samples need to pad to this size
    # print("A: ", A)
    A_2 = torch.bmm(A, A)
    A_3 = torch.bmm(A_2, A)
    tri_count = torch.diagonal(A_3, dim1=-2, dim2=-1).sum(1) / 6  # (B), triangle num of each sample,取对角线元素，以行求和
    # print("tri_count:", tri_count)
    triu_A = torch.triu(A)#返回上三角矩阵
    line_count = triu_A.sum((2, 1))#每一个0维上的12维求和
    # print("line_count:", line_count)
    count = (tri_count + line_count).long()
    max_num = torch.max(count).item()
    # print("count:", count)
    # print("max_num:", max_num)
    return count, max_num

def batch_line_num(A):  # calculate the max triangle number in batch, so other samples need to pad to this size
    # print("A: ", A)
    triu_A = torch.triu(A)
    line_count = triu_A.sum((2, 1))
    # print("line_count:", line_count)
    count = line_count.long()
    max_num = torch.max(count).item()
    # print("count:", count)
    # print("max_num:", max_num)
    return count, max_num


def generate_triangle_vertices_v2(keypoint, A):
    # print(A)
    Bn = A.size()[0]
    kp_num = A.size()[1]
    triu_A = torch.triu(A)#返回一个上三角矩阵
    count, max_num = batch_triangle_line_num(A)# calculate the max triangle number in batch, so other samples need to pad to this size
    vertices = torch.LongTensor([]).cuda()
    # print(count, max_num)
    '''generate triangle vertices'''
    edge_indices = triu_A.nonzero(as_tuple=False)  #用于输出数组的非零值的索引，即用来定位数组中非零的元素
    u = edge_indices[:, [0, 1]]  # first index
    v = edge_indices[:, [0, 2]]  # second index
    e = u * torch.Tensor([[1, kp_num]]).cuda() + v * torch.Tensor([[0, 1]]).cuda()  # edge index (u * kp_num + v) 对应行*行，列*列
    e = e[:, 1].long()#(u * kp_num + v)
    # u_e = torch.cat((u, e[:, 1].unsqueeze(1)), -1)
    # u_v = torch.cat((v, e[:, 1].unsqueeze(1)), -1)
    batch_index = u[:, 0].long()
    u_index = u[:, 1].long()#edge_indices的第二列
    v_index = v[:, 1].long()#edge_indices的第三列
    # print("batch_index: ", batch_index)
    edge_matrix = torch.zeros(Bn, kp_num, kp_num * kp_num).cuda()
    edge_matrix = edge_matrix.index_put([batch_index, u_index, e], torch.tensor(1.).cuda())
    edge_matrix = edge_matrix.index_put([batch_index, v_index, e], torch.tensor(1.).cuda())
    # print("edge_matrix: ", edge_matrix)
    final_matrix = torch.bmm(A, edge_matrix)
    tri_position = (final_matrix == 2).nonzero(as_tuple=False)  # return indices where items equal to 2.
    # print("tri_position: ", tri_position)
    u_origin = torch.div(tri_position[:, 2].float(), kp_num).floor().long()
    v_origin = torch.remainder(tri_position[:, 2], kp_num)#返回一个新张量，包含输入input张量每个元素的除法余数
    triangle_vertices = torch.cat((tri_position[:, :2], u_origin.unsqueeze(1), v_origin.unsqueeze(1)), 1)

    '''generate skeleton vertices, and expand (a,b) to (a,b,b) to fit the format of triangle vertices'''
    line_vertices = torch.cat((edge_indices, edge_indices[:, 2].unsqueeze(1)), -1)
    line_batch_info = line_vertices[:, 0]  # items represent batch num
    triangle_batch_info = triangle_vertices[:, 0]  # items represent batch num
    # print("info:", triangle_batch_info)

    for i in range(Bn):
        triangle_batch_i = (triangle_batch_info == i).nonzero(as_tuple=False).squeeze(-1)
        # print(triangle_batch_i)
        triangle_vertices_i = triangle_vertices[triangle_batch_i]  # triangle vertices of batch i
        # print("triangle_vertices: ", triangle_vertices_i.shape)
        line_batch_i = (line_batch_info == i).nonzero(as_tuple=False).squeeze(-1)
        line_vertices_i = line_vertices[line_batch_i]  # line vertices of batch i

        all_vertices_i = torch.cat((triangle_vertices_i[:, 1:], line_vertices_i[:, 1:]), 0)
        all_vertices_i, _ = torch.sort(all_vertices_i, dim=-1)
        all_vertices_i = all_vertices_i.unique(dim=0)
        # print("all_vertices_i:", all_vertices_i.shape)
        all_vertices_i = torch.cat(
            (all_vertices_i, torch.LongTensor([[0, 0, 0]]).cuda().expand(max_num - count[i].item(), 3)), 0)
        # print("all_vertices_i:", all_vertices_i.shape)
        vertices = torch.cat((vertices, all_vertices_i.unsqueeze(0)), 0)
    return vertices.long()


def generate_triangle_vertices_v3(keypoint, A):
    # print(A)
    Bn = A.size()[0]
    kp_num = A.size()[1]
    triu_A = torch.triu(A)
    count, max_num = batch_line_num(A)
    vertices = torch.LongTensor([]).cuda()
    # print(count, max_num)
    '''generate triangle vertices'''
    edge_indices = triu_A.nonzero(as_tuple=False)
    '''generate skeleton vertices, and expand (a,b) to (a,b,b) to fit the format of triangle vertices'''
    line_vertices = torch.cat((edge_indices, edge_indices[:, 2].unsqueeze(1)), -1)
    line_batch_info = line_vertices[:, 0]  # items represent batch num

    for i in range(Bn):

        line_batch_i = (line_batch_info == i).nonzero(as_tuple=False).squeeze(-1)
        line_vertices_i = line_vertices[line_batch_i]  # line vertices of batch i

        all_vertices_i = line_vertices_i[:, 1:]
        all_vertices_i, _ = torch.sort(all_vertices_i, dim=-1)
        all_vertices_i = all_vertices_i.unique(dim=0)

        all_vertices_i = torch.cat(
            (all_vertices_i, torch.LongTensor([[0, 0, 0]]).cuda().expand(max_num - count[i].item(), 3)), 0)
        # print("all_vertices_i:", all_vertices_i.shape)
        vertices = torch.cat((vertices, all_vertices_i.unsqueeze(0)), 0)
    return vertices.long()


def Batch_Keypoint_graph_interpolation_v22(batch_keypoint, graph):  # batch_keypoint: (B, k, 3)
    Bn = batch_keypoint.size()[0]
    batch_keypoint = batch_keypoint.cuda()
    vertices_index = generate_triangle_vertices_v2(batch_keypoint, graph)  # (B, max_num, 3)

    triangles = batch_keypoint[torch.arange(Bn).unsqueeze(-1).unsqueeze(-1), vertices_index]#
    return triangles


def Batch_Keypoint_graph_interpolation_v33(batch_keypoint, graph):  # Only Skeleton
    Bn = batch_keypoint.size()[0]
    vertices_index = generate_triangle_vertices_v3(batch_keypoint, graph)  # (B, max_num, 3)
    triangles = batch_keypoint[torch.arange(Bn).unsqueeze(-1).unsqueeze(-1), vertices_index]

    return triangles






if __name__ == '__main__':

    '''runtime test'''
    pc = torch.randn(32, 1024, 3).cuda()
    kp = torch.randn(32, 16, 3).cuda()
    # Batch_Keypoint_interpolation(test, 1024)

