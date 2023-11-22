# Hamiltonian-based Neural ODE Networks on the SE(3) Manifold For Dynamics Learning and Control, RSS 2021
# Thai Duong, Nikolay Atanasov

# code structure follows the style of HNN by Greydanus et al. and SymODEM by Zhong et al.
# https://github.com/greydanus/hamiltonian-nn
# https://github.com/Physics-aware-AI/Symplectic-ODENet

import torch
import numpy as np

from model import MLP, PSD, MatrixNet
from utils import compute_rotation_matrix_from_quaternion
from utils import L2_loss


class SE3HamNODEStruct(torch.nn.Module):
    def __init__(self, device=None, pretrain=True, M_net1=None, M_net2=None, D_net1=None, D_net2=None, V_net=None,
                 g_net=None, udim=2):
        super(SE3HamNODEStruct, self).__init__()
        init_gain = 0.001
        self.xdim = 3
        self.Rdim = 9
        self.linveldim = 3
        self.angveldim = 3
        self.posedim = self.xdim + self.Rdim  # 3 for position + 12 for rotmat
        self.twistdim = self.linveldim + self.angveldim  # 3 for linear vel + 3 for ang vel
        self.udim = udim
        self.diag_dim = 3
        self.off_diag_dim = 3
        if M_net1 is None:
            self.M_net1 = torch.nn.Parameter(torch.ones(self.xdim).to(device), requires_grad=True).to(device)
        else:
            self.M_net1 = M_net1
        if M_net2 is None:
            self.M_net2 = torch.nn.Parameter(torch.ones(self.xdim).to(device), requires_grad=True).to(device)
        else:
            self.M_net2 = M_net2
        if D_net1 is None:
            # self.D_net1 = torch.nn.Parameter(torch.zeros(2 * self.linveldim).to(device), requires_grad=True).to(device)
            self.D_net1 = PSD(self.posedim + self.twistdim, 400, self.linveldim, init_gain=init_gain).to(device)
        else:
            self.D_net1 = D_net1
        if D_net2 is None:
            # self.D_net2 = torch.nn.Parameter(torch.zeros(2 * self.angveldim).to(device), requires_grad=True).to(device)
            self.D_net2 = PSD(self.posedim + self.twistdim, 400, self.angveldim, init_gain=init_gain).to(device)
        else:
            self.D_net2 = D_net2
        if V_net is None:
            self.V_net = MLP(self.posedim, 400, 1, init_gain=init_gain).to(device)
        else:
            self.V_net = V_net
        if g_net is None:
            self.g_net = torch.nn.Parameter(torch.zeros(self.twistdim, self.udim).to(device), requires_grad=True).to(
                device)
        else:
            self.g_net = g_net

        self.device = device
        self.nfe = 0
        if pretrain:
            self.pretrain()

    def pretrain(self):
        print(f'NOTHING TO PRETRAIN')

    def forward(self, t, input):
        with torch.enable_grad():
            batch = input.shape[0]
            self.nfe += 1
            q, q_dot, u = torch.split(input, [self.posedim, self.twistdim, self.udim], dim=1)
            x, R = torch.split(q, [self.xdim, self.Rdim], dim=1)
            q_dot_v, q_dot_w = torch.split(q_dot, [self.linveldim, self.angveldim], dim=1)

            M_q_inv1 = torch.diag(self.M_net1).repeat(batch, 1, 1)
            M_q_inv2 = torch.diag(self.M_net2).repeat(batch, 1, 1)

            q_dot_aug_v = torch.unsqueeze(q_dot_v, dim=2)
            q_dot_aug_w = torch.unsqueeze(q_dot_w, dim=2)
            pv = torch.squeeze(torch.matmul(torch.inverse(M_q_inv1), q_dot_aug_v), dim=2)
            pw = torch.squeeze(torch.matmul(torch.inverse(M_q_inv2), q_dot_aug_w), dim=2)
            q_p = torch.cat((q, pv, pw), dim=1)
            q, pv, pw = torch.split(q_p, [self.posedim, self.linveldim, self.angveldim], dim=1)
            x, R = torch.split(q, [self.xdim, self.Rdim], dim=1)

            # Neural networks' forward passes
            V_q = self.V_net(q)
            g_q = torch.zeros(self.twistdim, self.udim).to(self.device)
            g_q[0, :] = self.g_net[0, :]
            g_q[-1, :] = self.g_net[-1, :]
            g_q = g_q.expand(batch, -1, -1)

            D_v = self.D_net1(q_p)
            D_w = self.D_net2(q_p)

            # # PSD
            # diag_v, off_diag_v = torch.split(self.D_net1, [self.diag_dim, self.off_diag_dim], dim=0)
            # Lv = torch.diag_embed(diag_v)
            # ind = np.tril_indices(self.diag_dim, k=-1)
            # flat_ind = np.ravel_multi_index(ind, (self.diag_dim, self.diag_dim))
            # Lv = torch.flatten(Lv, start_dim=0)
            # Lv[flat_ind] = off_diag_v
            # Lv = torch.reshape(Lv, (self.diag_dim, self.diag_dim))
            # Dv = torch.matmul(Lv, Lv.permute(1, 0))

            # diag_w, off_diag_w = torch.split(self.D_net2, [self.diag_dim, self.off_diag_dim], dim=0)
            # Lw = torch.diag_embed(diag_w)
            # ind = np.tril_indices(self.diag_dim, k=-1)
            # flat_ind = np.ravel_multi_index(ind, (self.diag_dim, self.diag_dim))
            # Lw = torch.flatten(Lw, start_dim=0)
            # Lw[flat_ind] = off_diag_w
            # Lw = torch.reshape(Lw, (self.diag_dim, self.diag_dim))
            # Dw = torch.matmul(Lw, Lw.permute(1, 0))
            # for i in range(self.diag_dim):
            #     Dv[i,i] = Dv[i,i] + 0.01
            #     Dw[i,i] = Dw[i,i] + 0.01
            # D_v = Dv.repeat(batch, 1, 1)
            # D_w = Dw.repeat(batch, 1, 1)

            # Calculate the Hamiltonian
            p_aug_v = torch.unsqueeze(pv, dim=2)
            p_aug_w = torch.unsqueeze(pw, dim=2)
            H = torch.squeeze(torch.matmul(torch.transpose(p_aug_v, 1, 2), torch.matmul(M_q_inv1, p_aug_v))) / 2.0 + \
                torch.squeeze(torch.matmul(torch.transpose(p_aug_w, 1, 2), torch.matmul(M_q_inv2, p_aug_w))) / 2.0 + \
                torch.squeeze(V_q)

            # Calculate the partial derivative using autograd
            dH = torch.autograd.grad(H.sum(), q_p, create_graph=True)[0]
            # Order: position (3), rotmat (9), lin vel (3) in body frame, ang vel (3) in body frame
            dHdx, dHdR, dHdpv, dHdpw = torch.split(dH, [self.xdim, self.Rdim, self.linveldim, self.angveldim], dim=1)

            # Calculate g*u
            F = torch.squeeze(torch.matmul(g_q, torch.unsqueeze(u, dim=2)), dim=2)

            # Hamilton's equation on SE(3) manifold for (q,p)
            Rmat = R.view(-1, 3, 3)
            dx = torch.squeeze(torch.matmul(Rmat, torch.unsqueeze(dHdpv, dim=2)), dim=2)
            dR03 = torch.cross(Rmat[:, 0, :], dHdpw)
            dR36 = torch.cross(Rmat[:, 1, :], dHdpw)
            dR69 = torch.cross(Rmat[:, 2, :], dHdpw)
            dR = torch.cat((dR03, dR36, dR69), dim=1)
            # dpv = torch.cross(pv, dHdpw) \
            #       - torch.squeeze(torch.matmul(torch.transpose(Rmat, 1, 2), torch.unsqueeze(dHdx, dim=2))) \
            #       + F[:, 0:3]
            # dpw = torch.cross(pw, dHdpw) \
            #       + torch.cross(pv, dHdpv) \
            #       + torch.cross(Rmat[:, 0, :], dHdR[:, 0:3]) \
            #       + torch.cross(Rmat[:, 1, :], dHdR[:, 3:6]) \
            #       + torch.cross(Rmat[:, 2, :], dHdR[:, 6:9]) \
            #       + F[:,3:6]
            dpv = torch.cross(pv, dHdpw) \
                  - torch.squeeze(torch.matmul(torch.transpose(Rmat, 1, 2), torch.unsqueeze(dHdx, dim=2)), dim=2) \
                  - torch.squeeze(torch.matmul(D_v, torch.unsqueeze(dHdpv, dim=2)), dim=2) + F[:, 0:3]
            dpw = torch.cross(pw, dHdpw) \
                  + torch.cross(pv, dHdpv) \
                  + torch.cross(Rmat[:, 0, :], dHdR[:, 0:3]) \
                  + torch.cross(Rmat[:, 1, :], dHdR[:, 3:6]) \
                  + torch.cross(Rmat[:, 2, :], dHdR[:, 6:9]) \
                  - torch.squeeze(torch.matmul(D_w, torch.unsqueeze(dHdpw, dim=2)), dim=2) + F[:, 3:6]

            # Hamilton's equation on SE(3) manifold for twist xi
            dM_inv_dt1 = torch.zeros_like(M_q_inv1)
            dv = torch.squeeze(torch.matmul(M_q_inv1, torch.unsqueeze(dpv, dim=2)), dim=2) \
                 + torch.squeeze(torch.matmul(dM_inv_dt1, torch.unsqueeze(pv, dim=2)), dim=2)

            dM_inv_dt2 = torch.zeros_like(M_q_inv2)
            dw = torch.squeeze(torch.matmul(M_q_inv2, torch.unsqueeze(dpw, dim=2)), dim=2) \
                 + torch.squeeze(torch.matmul(dM_inv_dt2, torch.unsqueeze(pw, dim=2)), dim=2)

            batch_size = input.shape[0]
            zero_vec = torch.zeros(batch_size, self.udim, dtype=torch.float32, device=self.device)
            return torch.cat((dx, dR, dv, dw, zero_vec), dim=1)