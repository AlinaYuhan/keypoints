
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import DropPath,trunc_normal_
#from pointnetpp.pointnet2_sem_seg_msg import get_model as PointNetPP1
from  .modeldgcnn import DGCNN_cls
from  .utils.triangle_utils import Batch_Keypoint_graph_interpolation_v33
from  .utils.DistFunc import distance,init_graph1,init_graph,pt2ske_graph
#from  anchor_points_test import GeoCrossAttention,Attention,Mlp


class PBlock(nn.Module):  # MLP Block
    def __init__(self, iu, *units, should_perm):
        super().__init__()
        self.sublayers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.should_perm = should_perm
        ux = iu
        for uy in units:
            self.sublayers.append(nn.Linear(ux, uy).cuda())
            self.batch_norms.append(nn.BatchNorm1d(uy).cuda())
            ux = uy

    def forward(self, input_x):
        x = input_x.cuda()
        for sublayer, batch_norm in zip(self.sublayers, self.batch_norms):
            x = sublayer(x)
            if self.should_perm:
                x = x.permute(0, 2, 1)
            x = batch_norm(x)
            if self.should_perm:
                x = x.permute(0, 2, 1)
            x = F.relu(x)
        return x


class Head(nn.Module):  # Decoder unit, one per line segment
    def __init__(self):
        super().__init__()
        self.emb = nn.Parameter(torch.randn((200, 3)) * 0.002).cuda()

    def forward(self, KPA, KPB):
        KPB = KPB.cuda()
        KPA = KPA.cuda()
        dist = torch.mean(torch.sqrt(1e-1 + (torch.sum(torch.square(KPA - KPB), dim=-1)))).cuda()
    #   dist = torch.mean(torch.sqrt(1e-3 + (torch.sum(torch.square(KPA - KPB), dim=-1)))).cuda()
        count = min(200, max(15, int((dist / 0.01).item())))
        device = dist.device
        #self.f_interp = torch.linspace(0.0, 1.0, count).unsqueeze(0).unsqueeze(-1).to(device)
        self.f_interp = torch.rand([1,count,1]).to(device)
        self.b_interp = 1.0 - self.f_interp

        x = KPA.unsqueeze(-2) * self.f_interp + KPB.unsqueeze(-2) * self.b_interp
        R = self.emb[:count, :].unsqueeze(0) + x  # N x count x 3
        return R.reshape((-1, count, 3)), self.emb


class Net1(nn.Module):
    def __init__(self, npt, k):#2048,10
        super().__init__()

        self.npt = npt
        self.k = k
        #self.PTW = PointNetPP1(k)
        self.PT_L = nn.Linear(k, k).cuda()
        self.DGC = DGCNN_cls(k)
        #self.grouper = DGCNN_group()
        #self.encoderblock = EncoderBlock(self.embed_dim,num_pred=16)
        self.MA = PBlock(1024,512,256, should_perm=False)
        self.MA_L = nn.Linear(256, k * (k - 1) // 2).cuda()
        self.DEC = nn.ModuleList()
        DECN = nn.ModuleList()
        DECN.append(Head())
        self.DEC.append(DECN)


        self.node_encoder = nn.Sequential(
            nn.Linear(3, k),
            nn.LeakyReLU(),
        ).cuda()

        self.edge_encoder = nn.Sequential(
            nn.Conv2d(k + k, k, kernel_size=1, bias=False),
            nn.LeakyReLU()
         ).cuda()

        self.degree_decoder = nn.Sequential(nn.Linear(1, 1, bias=True), nn.LeakyReLU()).cuda()



    def forward(self, input_x):

        dgx,gf = self.DGC(input_x.permute(0, 2, 1))
        dgx = self.PT_L(dgx)
        # #
        kp_heatmaps = F.softmax(dgx.permute(0, 2, 1), -1)  # [n, k, npt]
        kpcd = kp_heatmaps.bmm(input_x)# KeyPoint ClouD [n, k, 3]
        print(dgx.shape,input_x.shape)

        node  = self.node_encoder(kpcd)
        edge = distance(node,node)
        #
        edge_feat = self.edge_encoder(edge)
        edge_rank = edge_feat.permute(0, 3, 2, 1).sum(-1)  # [E]
        edge_rank = torch.sigmoid(edge_rank)
        srt_edge_rank, idxs = torch.sort(edge_rank, dim=-1, descending=True)  # 递True减排序
        c4 = init_graph1(input_x,kpcd,idxs)
        #c4 = init_graph(input_x, kpcd)

        ckp_coarse = Batch_Keypoint_graph_interpolation_v33(kpcd, c4)
        coarse =  ckp_coarse[:,:,:2,:]
        Bn = coarse.size()[1]
        #
        rp = []  # Reconstructed Parts
        ofs = []  # Offset embeddings for regularization loss
        for i in range(Bn):
                 aa = coarse[:,i]
                 R, EM = self.DEC[0][0](aa[:,0],aa[:,1])#self.DEC[i][j](aa[:,0],aa[:,1])
                 rp.append(R)
                 ofs.append(EM)


        rrp = torch.cat(rp, dim=1)
        strengths = F.sigmoid(self.MA_L(self.MA(gf)))
        #print(strengths.shape,rrp.shape)
        ofs = torch.cat(ofs, dim=1)  # P x 72 x 3

        return rp,kpcd, ofs,strengths



if __name__ == '__main__':

    net = Net1(2048,10).to('cuda')
    x = torch.randn((5,4048,3)).cuda().float()#.to(device)
    RPCD, KPCD, OFS,STR  = net(x)#LF, MA ,STR




