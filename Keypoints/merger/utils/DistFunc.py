import torch
import numpy as np


def distance(p1, p2):
    '''
    :param p1: size[B,N,D]
    :param p2: size[B,M,D]
    :param k: k nearest neighbors
    :param is_max: k-nearest neighbors or k-farthest neighbors
    :return: for each point in p1, returns the indices of the k nearest points in p2; size[B,N,k]
    '''
    assert p1.size(0) == p2.size(0) and p1.size(2) == p2.size(2)

    p1 = p1.unsqueeze(1)
    p2 = p2.unsqueeze(1)

    p1 = p1.repeat(1, p2.size(2), 1, 1)
    p1 = p1.transpose(1, 2)#每一行出现p2.size(2)次
    p2 = p2.repeat(1, p1.size(1), 1, 1)

    #dist = p1 - p2
    dist = torch.add(p1, torch.neg(p2))#torch.neg:按元素取负
    #dist = torch.norm(dist, 2, dim=3)
    feature = torch.cat((dist, p1), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature



def knn_with_batch(p1, p2, k, is_max=False):
    '''
    :param p1: size[B,N,D]
    :param p2: size[B,M,D]
    :param k: k nearest neighbors
    :param is_max: k-nearest neighbors or k-farthest neighbors
    :return: for each point in p1, returns the indices of the k nearest points in p2; size[B,N,k]
    '''
    assert p1.size(0) == p2.size(0) and p1.size(2) == p2.size(2)

    p1 = p1.unsqueeze(1)
    p2 = p2.unsqueeze(1)

    p1 = p1.repeat(1, p2.size(2), 1, 1)
    p1 = p1.transpose(1, 2)#每一行出现p2.size(2)次
    p2 = p2.repeat(1, p1.size(1), 1, 1)

    dist = torch.add(p1, torch.neg(p2))#torch.neg:按元素取负
    dist = torch.norm(dist, 2, dim=3)

    top_dist, k_nn = torch.topk(dist, k, dim=2, largest=is_max)

    return k_nn



def knn(p1, p2, k, is_max=False):
    '''
    :param p1: size[B,N,D]
    :param p2: size[B,M,D]
    :param k: k nearest neighbors
    :param is_max: k-nearest neighbors or k-farthest neighbors
    :return: for each point in p1, returns the indices of the k nearest points in p2; size[B,N,k]
    '''
    assert p1.size(0) == p2.size(0) and p1.size(2) == p2.size(2)

    p1 = p1.unsqueeze(1)
    p2 = p2.unsqueeze(1)

    p1 = p1.repeat(1, p2.size(2), 1, 1)
    p1 = p1.transpose(1, 2)#每一行出现p2.size(2)次
    p2 = p2.repeat(1, p1.size(1), 1, 1)

    dist = torch.add(p1, torch.neg(p2))#torch.neg:按元素取负
    dist = torch.norm(dist, 2, dim=3)

    top_dist, k_nn = torch.topk(dist, k, dim=2, largest=is_max)

    return top_dist,k_nn


def init_graph1(shape_xyz, skel_xyz,idx):
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

    A[torch.arange(bn)[:, None, None], torch.arange(pn)[None, :, None], idx[:, :, 1:3]] = 1
    A[torch.arange(bn)[:, None, None], idx[:, :, 1:3], torch.arange(pn)[None, :, None]] = 1

    #A[torch.arange(bn)[:, None, None], v, torch.arange(pn)[None, :, None]] = 1

    A = torch.triu(A, diagonal=1) + torch.triu(A.transpose(1, 2), diagonal=1).transpose(1, 2)
    B = torch.triu(A)

    for i in range(bn):
        b = B[i]
        edge_indices = b.nonzero(as_tuple=False)
        for j in range(edge_indices.size(0)):
            kp1_index = edge_indices[j, 0]; kp2_index = edge_indices[j, 1]
            kp1 = skel_xyz[i, kp1_index]; kp2 = skel_xyz[i, kp2_index]

            top_dist1, k_nn1 = knn(kp1.reshape([-1, 1, 3]), shape_xyz[i].unsqueeze(0), 1, is_max=False)
            knn_max1 = shape_xyz[i, k_nn1[:, :, 0]]

            top_dist2, k_nn2 = knn(kp2.reshape([-1, 1, 3]), shape_xyz[i].unsqueeze(0), 1, is_max=False)
            knn_max2 = shape_xyz[i, k_nn2[:, :, 0]]

            midd_max = (knn_max1 + knn_max2) / 2
            midd_kp = (kp1 + kp2) / 2
            midd_kp = midd_kp.reshape(midd_max.shape)
            diist, near_index = knn(midd_kp, shape_xyz[i].unsqueeze(0), 1, is_max=False)


            dist = diist.reshape([-1, 1])
            if dist > 0.15:
                B[i, kp1_index, kp2_index] = 0

    return B


def init_graph(shape_xyz, skel_xyz):
    bn, pn = skel_xyz.size()[0], skel_xyz.size()[1]

    knn_skel = knn_with_batch(skel_xyz, skel_xyz, pn, is_max=False)
    knn_sp2sk = knn_with_batch(shape_xyz, skel_xyz, 3, is_max=False)


    A = torch.zeros((bn, pn, pn)).float().cuda()

    # initialize A with recovery prior: Mark A[i,j]=1 if (i,j) are two skeletal points closest to a surface point
    A[torch.arange(bn)[:, None], knn_sp2sk[:, :, 0], knn_sp2sk[:, :, 1]] = 1
    A[torch.arange(bn)[:, None], knn_sp2sk[:, :, 1], knn_sp2sk[:, :, 0]] = 1

    # initialize A with topology prior
    A[torch.arange(bn)[:, None, None], torch.arange(pn)[None, :, None], knn_skel[:, :, 1:4]] = 1
    A[torch.arange(bn)[:, None, None], knn_skel[:, :, 1:4], torch.arange(pn)[None, :, None]] = 1


    A = torch.triu(A, diagonal=1) + torch.triu(A.transpose(1, 2), diagonal=1).transpose(1, 2)
    B = torch.triu(A)

    for i in range(bn):
          b = B[i]
          edge_indices = b.nonzero(as_tuple=False)
          for j in range(edge_indices.size(0)):
              kp1_index = edge_indices[j, 0]; kp2_index = edge_indices[j, 1]
              kp1 = skel_xyz[i, kp1_index]; kp2 = skel_xyz[i, kp2_index]
    # #
              top_dist1, k_nn1 = knn(kp1.reshape([-1, 1, 3]), shape_xyz[i].unsqueeze(0), 1, is_max=False)
              knn_max1 = shape_xyz[i, k_nn1[:, :, 0]]
    # #
              top_dist2, k_nn2 = knn(kp2.reshape([-1, 1, 3]), shape_xyz[i].unsqueeze(0), 1, is_max=False)
              knn_max2 = shape_xyz[i, k_nn2[:, :, 0]]
    # #
              midd_max = (knn_max1 + knn_max2) / 2
              midd_kp = (kp1 + kp2) / 2
              midd_kp = midd_kp.reshape(midd_max.shape)
              diist, near_index = knn(midd_kp, shape_xyz[i].unsqueeze(0), 1, is_max=False)
    # #
              dist = diist.reshape([-1, 1])
              if dist > 0.058:
                  B[i, kp1_index, kp2_index] = 0

    return B

def pt2ske_graph(shape_xyz, skel_xyz):
    bn, pn = skel_xyz.size()[0], skel_xyz.size()[1]

    knn_skel = knn_with_batch(skel_xyz, skel_xyz, pn, is_max=False)
    knn_sp2sk = knn_with_batch(shape_xyz, skel_xyz, 3, is_max=False)


    A = torch.zeros((bn, pn, pn)).float().cuda()

    # initialize A with recovery prior: Mark A[i,j]=1 if (i,j) are two skeletal points closest to a surface point
    A[torch.arange(bn)[:, None], knn_sp2sk[:, :, 0], knn_sp2sk[:, :, 1]] = 1
    A[torch.arange(bn)[:, None], knn_sp2sk[:, :, 1], knn_sp2sk[:, :, 0]] = 1

    # initialize A with topology prior
    A[torch.arange(bn)[:, None, None], torch.arange(pn)[None, :, None], knn_skel[:, :, 1:4]] = 1
    A[torch.arange(bn)[:, None, None], knn_skel[:, :, 1:3], torch.arange(pn)[None, :, None]] = 1

    A = torch.triu(A, diagonal=1) + torch.triu(A.transpose(1, 2), diagonal=1).transpose(1, 2)

    return A

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist








