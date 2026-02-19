
import torch
#from scipy.optimize import linear_sum_assignment
#from einops import repeat
import numpy as np

torch.square = lambda x: x ** 2
torch.minimum = lambda x, y: torch.min(torch.stack((x, y)), dim=0)[0]


def composed_sqrt_chamfer(y_true, y_preds, activations):
    L = 0.0
    # activations: N x P where P: # sub-clouds
    # y_true: N x ? x 3
    # y_pred: P x N x ? x 3 
    part_backs = []
    for i, y_pred in enumerate(y_preds):
        # y_true: k1 x 3
        # y_pred: k2 x 3
        y_true_rep = torch.unsqueeze(y_true, axis=-2)  # k1 x 1 x 3
        y_pred_rep = torch.unsqueeze(y_pred, axis=-3)  # 1 x k2 x 3
        # k1 x k2 x 3
        y_delta = torch.sqrt(1e-4 + torch.sum(torch.square(y_pred_rep - y_true_rep), -1))
        # k1 x k2
        y_nearest = torch.min(y_delta, -2)[0]
        # k2
        part_backs.append(torch.min(y_delta, -1)[0])
        L = L + torch.mean(torch.mean(y_nearest, -1) * activations[:, i]) / len(y_preds)
    part_back_stacked = torch.stack(part_backs)  # 增加新的维度进行堆叠 P x N x k1
    sorted_parts, indices = torch.sort(part_back_stacked, dim=0)#从小到大排序
    weights = torch.ones_like(sorted_parts[0])  # N x k1
    for i in range(len(y_preds)):
         w = torch.minimum(weights, torch.gather(activations, -1, indices[i]))
         L = L + torch.mean(sorted_parts[i] * w)
         weights = weights - w
    L = L + torch.mean(weights * 20.0)
    return L

def calc_cd(y_true, y_preds):
    pc_src_input = y_true.permute(0, 2, 1)
    pc_dst_input = y_preds.permute(0, 2, 1)

    B, M = pc_src_input.size()[0], pc_src_input.size()[2]
    N = pc_dst_input.size()[2]

    pc_src_input_expanded = pc_src_input.unsqueeze(3).expand(B, 3, M, N).cuda()
    pc_dst_input_expanded = pc_dst_input.unsqueeze(2).expand(B, 3, M, N).cuda()
    # the gradient of norm is set to 0 at zero-input. There is no need to use custom norm anymore.
    diff = torch.norm(pc_src_input_expanded- pc_dst_input_expanded, dim=1, keepdim=False)  # BxMxN

    # # pc_src vs selected pc_dst, M
    src_dst_min_dist, _ = torch.min(diff, dim=2, keepdim=False)  # BxM
    forward_loss = src_dst_min_dist.mean()

    # pc_dst vs selected pc_src, N
    dst_src_min_dist, _ = torch.min(diff, dim=1, keepdim=False)  # BxN
    backward_loss= dst_src_min_dist.mean()

    loss = (forward_loss+backward_loss)/2

    return loss

def compute_chamfer_distance(p1, p2):
    '''
    Calculate Chamfer Distance between two point sets
    :param p1: size[bn, N, D]
    :param p2: size[bn, M, D]
    :param debug: whether need to output debug info
    :return: sum of Chamfer Distance of two point sets
    '''

    diff = p1[:, :, None, :] - p2[:, None, :, :]
    dist = torch.sum(diff*diff,  dim=3)
    dist1 = dist
    dist2 = torch.transpose(dist, 1, 2)

    dist_min1, _ = torch.min(dist1, dim=2)
    dist_min2, _ = torch.min(dist2, dim=2)

    cd_p = (torch.sqrt(dist_min1 + 1e-14).mean(1) + torch.sqrt(dist_min2 + 1e-14).mean(1)) / 2
    cd_t = (dist_min1.mean(1) + dist_min2.mean(1))
    return dist_min1, dist_min2

def sample_farthest_points(points, num_samples, return_index=False):
    b, c, n = points.shape
    sampled = torch.zeros((b, 3, num_samples), device=points.device, dtype=points.dtype)
    indexes = torch.zeros((b, num_samples), device=points.device, dtype=torch.int64)

    index = torch.randint(n, [b], device=points.device)

    gather_index = repeat(index, 'b -> b c 1', c=c)
    sampled[:, :, 0] = torch.gather(points, 2, gather_index)[:, :, 0]
    indexes[:, 0] = index
    dists = torch.norm(sampled[:, :, 0][:, :, None] - points, dim=1)

    # iteratively sample farthest points
    for i in range(1, num_samples):
        _, index = torch.max(dists, dim=1)
        gather_index = repeat(index, 'b -> b c 1', c=c)
        sampled[:, :, i] = torch.gather(points, 2, gather_index)[:, :, 0]
        indexes[:, i] = index
        dists = torch.min(dists, torch.norm(sampled[:, :, i][:, :, None] - points, dim=1))

    if return_index:
        return sampled, indexes
    else:
        return sampled


def directed_hausdorff(point_cloud1: torch.Tensor, point_cloud2: torch.Tensor, reduce_mean=True):
    """
    :param point_cloud1: (B, 3, N)
    :param point_cloud2: (B, 3, M)
    :return: directed hausdorff distance, A -> B
    """
    n_pts1 = point_cloud1.shape[2]
    n_pts2 = point_cloud2.shape[2]

    pc1 = point_cloud1.unsqueeze(3)
    pc1 = pc1.repeat((1, 1, 1, n_pts2)).cuda() # (B, 3, N, M)
    pc2 = point_cloud2.unsqueeze(2)
    pc2 = pc2.repeat((1, 1, n_pts1, 1)).cuda()  # (B, 3, N, M)

    l2_dist = torch.sqrt(torch.sum((pc1 - pc2) ** 2, dim=1))  # (B, N, M)
    shortest_dist, _ = torch.min(l2_dist, dim=2)

    hausdorff_dist, _ = torch.max(shortest_dist, dim=1)  # (B, )

    if reduce_mean:
        hausdorff_dist = torch.mean(hausdorff_dist)

    return hausdorff_dist

def EMD(x, y, dist=None, return_dist=False):
    """ Computing earth mover distance bewteen two batch of points
    :param x:  (b, n, dim)
    :param y:  (b, m, dim)
    :param dist: (b, n, m) precomputed pairwise distances
    :param return_dist: Whether return the pairwise distances
    :return: (b,) Batch of EMDs
    """

    b, n, m, dim = x.size(0), x.size(1), y.size(1), x.size(2)
    #assert n == m, "EMD only works if two point clouds are equal size"
    if dist is None:
            dim = x.shape[-1]
            x = x.reshape(b, n, 1, dim).cuda()
            y = y.reshape(b, 1, m, dim).cuda()
            dist = (x - y).norm(dim=-1, keepdim=False)  # (b, n, m)

    emd_lst = []
    dist_np = dist.cpu().detach().numpy()
    for i in range(b):
            d_i = dist_np[i]
            r_idx, c_idx = linear_sum_assignment(d_i)
            emd_i = d_i[r_idx, c_idx].mean()
            emd_lst.append(emd_i)
    emd = np.stack(emd_lst).reshape(-1)
    emd_torch = torch.from_numpy(emd).to(x)
    if return_dist:
            return emd_torch, dist
    return emd_torch, None

def overlap_loss_torch_error(kp, threshold=0.05):
    '''
    Parameters
    ----------
    kp:         Key-points
    threshold   allowable overlap between the key-points
    Method:     Find distance of every point from all the points
                select the minimum distances that are greater than 0 (distance from itself)
                return count of the separated distances => final loss

    Returns     separation loss -> avoid estimation of multiple key-points on the same 3D location
    -------
    '''

    distances = torch.cat([torch.squeeze(
        torch.norm(kp[i].unsqueeze(1) - kp[i].unsqueeze(0), dim=2, p=None)) for i in range(len(kp))], dim=0)

    return torch.count_nonzero(distances[(distances < threshold)] >0) / len(kp)*len(kp)

