import torch
import torch.nn as nn
from torch.nn import *
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from settings import *
import torch.utils.data as Data
import math

class GATs(nn.Module):
    def __init__(self, args, query=False, padding=ord(' ')):
        super(GATs, self).__init__()
        self.query = query
        self.args = args

        self.device = torch.device(self.args.device)
        self.rel_attn = BatchMultiHeadGraphAttention(self.device, self.args)
        self.rel_mlp = nn.Sequential(
            nn.Linear(LaBSE_DIM * 2, LaBSE_DIM * 2, bias=False)
        )

        self.mattn = nn.Parameter(torch.ones(3, 10), requires_grad=True)
        self.gattn = nn.Parameter(torch.ones(3, 10), requires_grad=True)
        if self.args.att:
            self.att_attn = VanillaGraphAttention(self.device, self.args)
            self.att_mlp = nn.Sequential(
                nn.Linear(LaBSE_DIM * 2, LaBSE_DIM * 2, bias=False)
            )


    def update(self, network: nn.Module):
        for key_param, query_param in zip(self.parameters(), network.parameters()):
            key_param.data *= self.args.momentum
            key_param.data += (1 - self.args.momentum) * query_param.data
        self.eval()

    def nei_agg(self, batch, adj):
        batch = batch.to(self.device)
        adj = adj.to(self.device)
        center = batch[:, 0]
        center_neigh = batch

        for i in range(0, self.args.gat_num):
            center_neigh = self.rel_attn(center_neigh, adj.bool()).squeeze(1)
        center_neigh = center_neigh[:, 0]
        if self.args.center_norm:
            center = F.normalize(center, p=2, dim=1)
        if self.args.neighbor_norm:
            center_neigh = F.normalize(center_neigh, p=2, dim=1)
        if self.args.combine:
            nei_hat = torch.cat((center, center_neigh), dim=1)
            nei_hat = self.rel_mlp(nei_hat)
            if self.args.emb_norm:
                nei_hat = F.normalize(nei_hat, p=2, dim=1)
        else:
            nei_hat = center_neigh
        return nei_hat

    def att_agg(self, att_batch, att_adj):
        att_batch = att_batch.to(self.device)
        att_adj = att_adj.to(self.device)
        att_center = att_batch[:, 0]
        att_neigh = att_batch
        for i in range(0, self.args.gat_num):
            att_neigh = self.att_attn(att_neigh)
        #att_neigh = att_neigh[:, 0]
        if self.args.center_norm:
            att_center = F.normalize(att_center, p=2, dim=1)
        if self.args.neighbor_norm:
            att_neigh = F.normalize(att_neigh, p=2, dim=1)
        if self.args.combine:
            att_hat = torch.cat((att_center, att_neigh), dim=1)
            att_hat = self.att_mlp(att_hat)
            # att_hat = att_center + att_neigh
            if self.args.emb_norm:
                att_hat = F.normalize(att_hat, p=2, dim=1)
        else:
            att_hat = att_neigh
        return att_hat

    def forward(self, batch, adj, att_batch, att_adj):
        nei_hat = self.nei_agg(batch, adj)

        if self.args.att:
            att_hat = self.att_agg(att_batch, att_adj)
            #w = F.softmax(self.fusion, dim=0)
            out_hat = torch.cat((nei_hat, att_hat), dim=1)
            out_hat = F.normalize(out_hat, p=2, dim=1)
            #out_hat = F.normalize(self.out_mlp(out_hat), p=2, dim=1)
            return out_hat, nei_hat, att_hat
        else:
            out_hat = nei_hat
        return out_hat, nei_hat, nei_hat


class BatchMultiHeadGraphAttention(nn.Module):
    def __init__(self, device, args, n_head=MULTI_HEAD_DIM, f_in=LaBSE_DIM, f_out=LaBSE_DIM, bias=True):
        super(BatchMultiHeadGraphAttention, self).__init__()
        self.device = device
        self.n_head = n_head
        self.w = Parameter(torch.Tensor(n_head, f_in, f_out))
        self.a_src = Parameter(torch.Tensor(n_head, f_out, 1))
        self.a_dst = Parameter(torch.Tensor(n_head, f_out, 1))

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(args.dropout)
        if bias:
            self.bias = Parameter(torch.Tensor(f_out))
            nn.init.constant_(self.bias, 0)
        else:
            self.register_parameter('bias', None)

        nn.init.xavier_uniform_(self.w)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)

    def forward(self, h, adj):
        bs, n = h.size()[:2]  # h is of size bs x n x f_in
        h_prime = torch.matmul(h.unsqueeze(1), self.w)  # bs x n_head x n x f_out
        attn_src = torch.matmul(torch.tanh(h_prime), self.a_src)  # bs x n_head x n x 1
        attn_dst = torch.matmul(torch.tanh(h_prime), self.a_dst)  # bs x n_head x n x 1
        attn = attn_src.expand(-1, -1, -1, n) + attn_dst.expand(-1, -1, -1, n).permute(0, 1, 3,
                                                                                       2)  # bs x n_head x n x n

        attn = self.leaky_relu(attn)
        mask = ~(adj.unsqueeze(1) | torch.eye(adj.shape[-1]).bool().to(self.device))  # bs x 1 x n x n
        attn.data.masked_fill_(mask, float("-inf"))
        attn = self.softmax(attn)  # bs x n_head x n x n
        #attn = self.dropout(attn)
        # logging.info("attn: ", attn)
        # logging.info("attn.shape: ", attn.shape)
        output = torch.matmul(attn, h_prime)  # bs x n_head x n x f_out
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class VanillaGraphAttention(nn.Module):
    def __init__(self, device, args, n_head=MULTI_HEAD_DIM, f_in=LaBSE_DIM, f_out=LaBSE_DIM, bias=True):
        super(VanillaGraphAttention, self).__init__()
        self.device = device
        self.args =args
        self.sqrt_dk = math.sqrt(f_in)
        self.w = Parameter(torch.Tensor(n_head, f_in, f_out))
        #self.b = Parameter(torch.Tensor(f_out))
        self.attn = Parameter(torch.Tensor(f_out, 1))
        self.softmax = nn.Softmax(dim=-1)
        self.relu = nn.LeakyReLU(negative_slope=0.2)
        self.dropout = nn.Dropout(args.dropout)
        init.xavier_uniform_(self.w)
        #init.constant_(self.b, 0)
        init.xavier_uniform_(self.attn)

    def forward(self, h):
        h_prime = (torch.matmul(h.unsqueeze(1), self.w)).squeeze(1)
        attn_beta = self.relu(torch.matmul(torch.tanh(h_prime), self.attn)).squeeze(2)
        attn_beta = torch.where(attn_beta == 0, float('-inf'), attn_beta)
        attn_beta = self.softmax(attn_beta).unsqueeze(1)
        #attn_beta = self.dropout(attn_beta)
        output = torch.matmul(attn_beta, h_prime).squeeze(1)
        return output

class VoteAttention(nn.Module):
    def __init__(self):
        super(VoteAttention, self).__init__()
        self.w = nn.Parameter(torch.ones((3, 3)),
                                   requires_grad = True)

    def forward(self, u, r, a):
        sim_u = torch.cat((u[:, :2] * self.w[0, :2], u[:, 2:] * self.w[0, 2:]), dim=1)
        sim_r = torch.cat((r[:, :2] * self.w[1, :2], r[:, 2:] * self.w[1, 2:]), dim=1)
        sim_a = torch.cat((a[:, :2] * self.w[2, :2], a[:, 2:] * self.w[2, 2:]), dim=1)
        return sim_u, sim_r, sim_a

class nsmf(nn.Module):
    def __init__(self, device, args, f_in=LaBSE_DIM, f_out=LaBSE_DIM*2):
        super(nsmf, self).__init__()
        self.device = device
        self.args = args
        self.fc = nn.Sequential(
            nn.Linear(f_in, f_out),
            nn.LazyBatchNorm1d()
        )
    def forward(self, h):
        out = self.fc(h)
        return out
