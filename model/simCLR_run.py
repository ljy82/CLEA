import logging
import time
import torch
import torch.nn as nn
import faiss
import math
from torch.nn import *
import torch.nn.functional as F
from torch.nn.functional import cosine_similarity
import numpy as np
import torch.optim as optim
from settings import *
import torch.utils.data as Data
from operator import itemgetter
from collections import Counter
from model.GATs_layers import *
from loader.DBP15K_DataSet import *
from loader.DWY100K_DataSet import *
from loader.WK31_DataSet import *
from model.loss import *


def adjust_learning_rate(optimizer, epoch, lr):
    if (epoch + 1) % 10 == 0:
        lr *= 0.5
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



##distance##################
def mypair_distance_min(a, b, distance_type="L2"):
    if distance_type == "L1":
        return functional.pairwise_distance(a, b, p=1)  # [B*C]
    elif distance_type == "L2":
        return functional.pairwise_distance(a, b, p=2)
    elif distance_type == "L2squared":
        return torch.pow(functional.pairwise_distance(a, b, p=2), 2)
    elif distance_type == "cosine":
        return 1 - torch.cosine_similarity(a, b)  # [B*C]

def drop_nei_feature(x, drop_prob):
    drop_mask = torch.empty(
        (x.size(1)),
        dtype=torch.float32,
        device=x.device).uniform_(0, 1) < drop_prob
    x = x.clone()
    x[:, drop_mask] = 0
    return x

def drop_att_feature(x, drop_prob):
    drop_mask = torch.empty(
        (x.size(1), x.size(2)),
        dtype=torch.float32,
        device=x.device).uniform_(0, 1) < drop_prob
    drop_mask[0, :] = False
    x = x.clone()
    x[:, drop_mask] = 0
    return x


def remove_adj(adj, drop_prob):
    mask = (adj[:, 0, :] == 1) & (torch.rand(adj.size(0), adj.size(2)) < drop_prob)
    adj[:, 0, :][mask] = 0
    adj[:, :, 0][mask] = 0
    return adj


def reliable_views(x, x_id, x_reinfo, pm, pt):
    x_id_new = x_id[:, 1:]
    mask = torch.index_select(x_reinfo, 0, x_id_new.flatten()).reshape(x_id_new.shape)
    mask = mask /mask.mean() * pm
    dorp_mask = mask.where(mask < pm, torch.ones_like(mask) * 1)
    sel_mask = torch.bernoulli(1 - dorp_mask).to(torch.bool)

    att_mask_tmp = sel_mask.unsqueeze(-1).expand(x.size(0), x.size(1)-1, x.size(-1))
    att_mask = att_mask_tmp.clone()

    random_tensor = torch.rand(att_mask.size()) < pt
    att_mask.masked_fill_(random_tensor, False)


    tmp = torch.zeros((x.size(0), 1, x.size(-1))).bool()
    att_mask = torch.cat((tmp, att_mask), dim=1)
    x = x.clone()
    x[att_mask] = 0
    return x

# 自定义KL散度正则化器
class KLDivergenceRegularizer(nn.Module):
    def __init__(self, args, weight=1.0):
        super(KLDivergenceRegularizer, self).__init__()
        self.target_distribution = self.custom_t_distribution(args)
        self.weight = weight

    def forward(self, weights):
        # 计算KL散度
        kl_divergence = F.kl_div(F.log_softmax(weights, dim=1), self.target_distribution, reduction='sum')
        return self.weight * kl_divergence

    def custom_t_distribution(self, args):

        t = torch.randint(1, 10, (3, 10), device=args.device)
        t = 0.1 * t
        sorted_tensor, _ = torch.sort(t, dim=1, descending=True)
        return F.softmax(sorted_tensor, dim=1)

def h1_loss(args, z1, neg):
    def sim(z1, z2):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    f = lambda x: torch.exp(x / args.t)
    between_sim = f(sim(z1, neg[:, 0, :].squeeze(1)))
    p = f(torch.eye(z1.size(0), device=z1.device))
    neg_sim = f(torch.bmm(neg[:, 1:, :], z1.unsqueeze(2)).squeeze(-1))

    return -torch.log(between_sim.diag() / (neg_sim.sum(1) + between_sim.sum(1) - between_sim.diag()))

def h2_loss(args, w, logits):
    f = lambda x: torch.exp(x / args.t)
    P = f(logits[:, 0])
    N = f(logits)
    loss = -torch.log(P/N.sum(1))
    return loss


def hard_neg_topk(language, batch, train_s, train_t, topk, model, method='cross'):

    m = 0.8
    def find_idx(z1, z2, k):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        sim = torch.matmul(z2, z1.t())
        simx, idx = torch.topk(sim, dim=0, k=k)
        return simx.t(), idx.t()


    def retopk(z1, z2, k, m):
        sim_u, idx_u = find_idx(z1[0], z2[0], k)
        sim_r, idx_r = find_idx(z1[1], z2[1], k)
        sim_a, idx_a = find_idx(z1[2], z2[2], k)
        mask1 = (idx_u.unsqueeze(2) == idx_r.unsqueeze(1))
        mask2 = (idx_a.unsqueeze(2) == idx_r.unsqueeze(1))
        w = model.gattn
        sim_rh = (sim_r * w[0, :])
        sim_ah = (sim_a * w[1, :])
        sim_uh = (sim_u * w[2, :])
        r_QKV = F.softmax(sim_rh, dim=-1) * sim_r
        a_QKV = F.softmax(sim_ah, dim=-1) * sim_a
        u_QKV = F.softmax(sim_uh, dim=-1) * sim_u
        sim_ut = (u_QKV.unsqueeze(2) * mask1).sum(dim=1)
        sim_at = (a_QKV.unsqueeze(2) * mask2).sum(dim=1)
        z_u = (sim_ut == 0)
        z_a = (sim_at == 0)
        sim_ut[z_u] = u_QKV.mean(dim=1, keepdim=True).expand(sim_u.size(0), k)[z_u]
        sim_at[z_a] = a_QKV.mean(dim=1, keepdim=True).expand(sim_a.size(0), k)[z_a]
        r = r_QKV + sim_ut + sim_at
        sorted_indices = torch.argsort(r, dim=1, descending=True)
        sorted_idu = torch.gather(idx_u, 1, sorted_indices)
        h1 = z2[0][sorted_idu]
        h2 = r.gather(dim=1, index=sorted_indices)
        h1[:, 1:, :] = m * h1[:, 1:, :] + (1-m) * h1[:, 0, :].unsqueeze(1)
        h2[:, 1:] = m * h2[:, 1:] + (1-m) * h2[:, 0].unsqueeze(1)
        h_train = [h1, h2]
        return h_train


    if method == 'cross':
        if language == '1':
            train = train_t
        else:
            train = train_s
    else:
        if language == '1':
            train = train_s
        else:
            train = train_t

    h_train = retopk(batch, train, topk, m)
    return h_train


def train_fn(epoch, args, traindata, qencoder, kencoder, criterion, optimizer, device, ent_reinfo1, ent_reinfo2, attr_reinfo1, attr_reinfo2, kl_regularizer):
    adjust_learning_rate(optimizer, epoch, args.lr)
    loss_total = 0
    start_time = time.time()
    g_u1, g_u2, g_n1, g_n2, g_a1, g_a2 = [], [], [], [], [], []
    for batch_id, (language, nei, ent_id, nei_adj, id_data, att, att_id, att_adj) in tqdm(enumerate(traindata)):
        with torch.no_grad():
            qencoder.eval()
            u, n_h, a_h = qencoder(nei, nei_adj, att, att_adj)
        if language == '1':
            g_u1.append(u)
            g_n1.append(n_h)
            g_a1.append(a_h)
        else:
            g_u2.append(u)
            g_n2.append(n_h)
            g_a2.append(a_h)
    g_u1, g_u2, g_n1, g_n2, g_a1, g_a2 = torch.cat(g_u1, dim=0), torch.cat(g_u2, dim=0), torch.cat(g_n1, dim=0), torch.cat(g_n2, dim=0), torch.cat(g_a1, dim=0), torch.cat(g_a2, dim=0)
    train_s = [g_u1, g_n1, g_a1]
    train_t = [g_u2, g_n2, g_a2]
    for batch_id, (language, nei, ent_id, nei_adj, id_data, att, att_id, att_adj) in tqdm(enumerate(traindata)):
        if args.gda:
            nei1 = drop_nei_feature(nei, args.drop_rate)
            nei2 = drop_nei_feature(nei, 0.3)


            if language == '1':

                att1 = reliable_views(att, att_id, attr_reinfo1, 0.3, 0.9)
                att2 = reliable_views(att, att_id, attr_reinfo1, 0.4, 0.9)
            else:

                att1 = reliable_views(att, att_id, attr_reinfo2, 0.3, 0.9)
                att2 = reliable_views(att, att_id, attr_reinfo2, 0.4, 0.9)

        else:
            nei1 = nei
            nei2 = nei
            att1 = att
            att2 = att



        u, n_h, a_h = qencoder(nei, nei_adj, att, att_adj)
        u1, n_h1, a_h1 = qencoder(nei1, nei_adj, att1, att_adj)
        with torch.no_grad():
            kencoder.eval()
            u2, n_h2, a_h2 = kencoder(nei2, nei_adj, att2, att_adj)
        loss1 = criterion(n_h1, n_h2, epoch) + criterion(n_h, n_h1, epoch)
        loss2 = criterion(a_h1, a_h2, epoch) + criterion(a_h, a_h1, epoch)
        loss = loss1 + loss2


        # ===================hard mining=====================
        if args.hard:
            batch = [u, n_h, a_h]
            neg = hard_neg_topk(language, batch, train_s, train_t, topk=10, model=qencoder, method='cross')
            h_loss = h1_loss(args, u, neg[0]) + h2_loss(args, u, neg[1])
            c_loss = kl_regularizer(qencoder.gattn)
            c_loss = c_loss.to(h_loss.device)
            loss = (loss + h_loss + c_loss).mean()
        else:
            loss = loss.mean()
        # ===================backward=====================

        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        kencoder.update(qencoder)
        loss_total = loss_total + loss.detach()

    logging.info('epoch: {} time:{}'.format(epoch, (time.time() - start_time)))


def evaluate_fn_cpu(epoch, args, evaldata, qencoder, kencoder):
    logging.info("Evaluate at epoch {}...".format(epoch))

    ids_1, ids_2, vector_1, vector_2 = list(), list(), list(), list()

    inverse_ids_2 = dict()
    with torch.no_grad():
        qencoder.eval()
        for sample_id_1, (batch, ent_id, adj, id_data, att_batch, att_id, att_adj) in tqdm(enumerate(evaldata[0])):
            entity_vector_1, r, _ = qencoder(batch, adj, att_batch, att_adj)
            entity_vector_1 = entity_vector_1.squeeze().detach().cpu().numpy()
            ids_1.extend(id_data.squeeze().tolist())
            vector_1.append(entity_vector_1)


        for sample_id_2, (batch, ent_id, adj, id_data, att_batch, att_id, att_adj) in tqdm(enumerate(evaldata[1])):
            entity_vector_2, r, _ = qencoder(batch, adj, att_batch, att_adj)
            entity_vector_2 = entity_vector_2.squeeze().detach().cpu().numpy()
            ids_2.extend(id_data.squeeze().tolist())
            vector_2.append(entity_vector_2)



    for idx, _id in enumerate(ids_2):
        inverse_ids_2[_id] = idx

    def calculate_mrr(predictions, labels):
        num_queries = predictions.shape[0]
        mrr_sum = 0.0
        for i in range(num_queries):
            label = labels[i]
            prediction_ranks = np.where(predictions[i] == label)[0]

            if len(prediction_ranks) > 0:
                prediction_rank = prediction_ranks[0] + 1
                mrr_sum += 1.0 / prediction_rank
        mrr = mrr_sum / num_queries
        return mrr

    def cal_hit(v1, v2, link):
        source = [_id for _id in ids_1 if _id in link]
        target = np.array(
            [inverse_ids_2[link[_id]] if link[_id] in inverse_ids_2 else 99999 for _id in source])
        src_idx = [idx for idx in range(len(ids_1)) if ids_1[idx] in link]
        v1 = np.concatenate(tuple(v1), axis=0)[src_idx, :]
        v2 = np.concatenate(tuple(v2), axis=0)
        index = faiss.IndexFlatL2(v2.shape[1])
        index.add(np.ascontiguousarray(v2))
        D, I = index.search(np.ascontiguousarray(v1), 10)
        hit1 = (I[:, 0] == target).astype(np.int32).sum() / len(source)
        hit10 = (I == target[:, np.newaxis]).astype(np.int32).sum() / len(source)
        tt = target[:, np.newaxis]
        mrr = calculate_mrr(I, tt)

        logging.info("#Entity: {}".format(len(source)))
        logging.info("Hit@1: {}".format(round(hit1, 3)))
        logging.info("Hit@10:{}".format(round(hit10, 3)))
        logging.info("MRR:{}".format(round(mrr, 3)))
        return round(hit1, 3), round(hit10, 3), round(mrr, 3)

    logging.info('========Validation========')
    hit1_valid, hit10_valid, mrr_valid = cal_hit(vector_1, vector_2, evaldata[2])
    logging.info('===========Test===========')
    hit1_test, hit10_test, mrr_test = cal_hit(vector_1, vector_2, evaldata[3])
    return hit1_valid, hit10_valid, hit1_test, hit10_test, mrr_valid, mrr_test



def evaluate_fn_gpu(epoch, args, evaldata, qencoder, kencoder):
    logging.info("Evaluate at epoch {}...".format(epoch))

    ids_1, ids_2, vector_1, vector_2 = list(), list(), list(), list()
    inverse_ids_2 = dict()
    with torch.no_grad():
        qencoder.eval()
        for sample_id_1, (batch, adj, id_data, att_batch, att_adj) in tqdm(enumerate(evaldata[0])):
            if args.att:
                entity_vector_1, _, _ = qencoder(batch, adj, att_batch, att_adj)
            else:
                entity_vector_1, _ = qencoder(batch, adj, att_batch, att_adj)
            entity_vector_1 = entity_vector_1.squeeze().detach()
            ids_1.extend(id_data.squeeze().tolist())
            vector_1.append(entity_vector_1)

        for sample_id_2, (batch, adj, id_data, att_batch, att_adj) in tqdm(enumerate(evaldata[1])):
            if args.att:
                entity_vector_2, _, _ = qencoder(batch, adj, att_batch, att_adj)
            else:
                entity_vector_2, _ = qencoder(batch, adj, att_batch, att_adj)
            entity_vector_2 = entity_vector_2.squeeze().detach()
            ids_2.extend(id_data.squeeze().tolist())
            vector_2.append(entity_vector_2)

    for idx, _id in enumerate(ids_2):
        inverse_ids_2[_id] = idx

    def cal_hit(v1, v2, link):
        source = [_id for _id in ids_1 if _id in link]
        target = [inverse_ids_2[link[_id]] if link[_id] in inverse_ids_2 else 99999 for _id in source]
        src_idx = [idx for idx in range(len(ids_1)) if ids_1[idx] in link]
        v1 = torch.concat(tuple(v1), dim=0)[src_idx, :]
        v2 = torch.concat(tuple(v2), dim=0)
        res = faiss.StandardGpuResources()
        index = faiss.IndexFlatL2(v2.shape[1])
        gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
        gpu_index.add(v2)
        D, I = gpu_index.search(v1, 10)
        hit1 = (I[:, 0] == target).type(torch.int32).sum() / len(source)
        hit10 = (I == target[:, np.newaxis]).type(torch.int32).sum() / len(source)
        logging.info("#Entity: {}".format(len(source)))
        logging.info("Hit@1: {}".format(round(hit1, 3)))
        logging.info("Hit@10:{}".format(round(hit10, 3)))
        return round(hit1, 3), round(hit10, 3)

    logging.info('========Validation========')
    hit1_valid, hit10_valid = cal_hit(vector_1, vector_2, evaldata[2])
    logging.info('===========Test===========')
    hit1_test, hit10_test = cal_hit(vector_1, vector_2, evaldata[3])
    return hit1_valid, hit10_valid, hit1_test, hit10_test



class main_loop(object):
    def __init__(self, args, seed=37):
        self.seed = seed
        fix_seed(seed)
        self.args = args
        # load raw data
        self.device = torch.device(args.device)
        if args.dataset == 'DBP15K':
            self.loader1, self.loader2, self.eval_loader1, self.eval_loader2, self.val_link, self.test_link, self.ent_reinfo1, self.ent_reinfo2, self.attr_reinfo1, self.attr_reinfo2 = dbp15kdataset(
                self.args).getdata()
        elif args.dataset == 'WK31':
            self.loader1, self.loader2, self.eval_loader1, self.eval_loader2, self.val_link, self.test_link, self.ent_reinfo1, self.ent_reinfo2, self.attr_reinfo1, self.attr_reinfo2 = wk31dataset(
                self.args).getdata()
        elif args.dataset == 'DWY100K':
            self.loader1, self.loader2, self.eval_loader1, self.eval_loader2, self.val_link, self.test_link, self.ent_reinfo1, self.ent_reinfo2, self.attr_reinfo1, self.attr_reinfo2 = dwy100kdataset(
                self.args).getdata()
        self.traindata = []

        self.evaldata = []
        self.evaldata.append(self.eval_loader1)
        self.evaldata.append(self.eval_loader2)
        self.evaldata.append(self.val_link)
        self.evaldata.append(self.test_link)
        for batch_id, (nei, ent_id, nei_adj, id_data, att, att_id, att_adj) in enumerate(self.loader1):
            self.traindata.append(['1', nei, ent_id, nei_adj, id_data, att, att_id, att_adj])
        for batch_id, (nei, ent_id, nei_adj, id_data, att, att_id, att_adj) in enumerate(self.loader2):
            self.traindata.append(['2', nei, ent_id, nei_adj, id_data, att, att_id, att_adj])
        del self.loader1, self.loader2
        logging.info("begin")


        # model
        self.qencoder = GATs(self.args, VOCAB_SIZE).to(self.device)
        self.kencoder = GATs(self.args, VOCAB_SIZE).to(self.device)
        self.kencoder.update(self.qencoder)
        self.kl_regularizer = KLDivergenceRegularizer(self.args).to(self.device)
        logging.info(self.qencoder)
        logging.info(self.kencoder)
        self.criterion = CL_Loss(self.args, self.device)
        params = [{'params': self.qencoder.parameters()}]
        self.optimizer = optim.Adam(params=params, lr=self.args.lr)

    def train(self):
        best_hit1_valid_epoch = 0
        best_hit10_valid_epoch = 0
        best_hit1_valid_hit10 = 0
        best_hit10_valid_hit1 = 0
        best_hit1_valid = 0
        best_hit10_valid = 0
        best_hit1_test = 0
        best_hit10_test = 0
        best_hit1_test_hit10 = 0
        best_hit10_test_hit1 = 0
        best_hit1_test_epoch = 0
        best_hit10_test_epoch = 0
        record_hit1 = 0
        record_hit10 = 0
        record_mrr = 0
        record_epoch = 0
        logging.info("*** Training  begining ***")
        fix_seed(self.seed)

        for epoch in range(self.args.epoch):
            train_fn(epoch, self.args, self.traindata, self.qencoder, self.kencoder, self.criterion, self.optimizer, self.device, self.ent_reinfo1, self.ent_reinfo2, self.attr_reinfo1, self.attr_reinfo2, self.kl_regularizer)
            hit1_valid, hit10_valid, hit1_test, hit10_test, mrr_valid, mrr_test = evaluate_fn_cpu(epoch, self.args, self.evaldata, self.qencoder,
                                                                         self.kencoder)
            if hit1_valid > best_hit1_valid:
                best_hit1_valid = hit1_valid
                best_hit1_valid_hit10 = hit10_valid
                best_hit1_valid_epoch = epoch
                record_epoch = epoch
                record_hit1 = hit1_test
                record_hit10 = hit10_test
                record_mrr = mrr_test
            if hit10_valid > best_hit10_valid:
                best_hit10_valid = hit10_valid
                best_hit10_valid_hit1 = hit1_valid
                best_hit10_valid_epoch = epoch
                if hit1_valid == best_hit1_valid:
                    record_epoch = epoch
                    record_hit1 = hit1_test
                    record_hit10 = hit10_test
                    record_mrr = mrr_test
            if hit1_test > best_hit1_test:
                best_hit1_test = hit1_test
                best_hit1_test_hit10 = hit10_test
                best_hit1_test_epoch = epoch
            if hit10_test > best_hit10_test:
                best_hit10_test = hit10_test
                best_hit10_test_hit1 = hit1_test
                best_hit10_test_epoch = epoch
            logging.info('Test Hit@1(10)    = {}({}) at epoch {} '.format(hit1_test, hit10_test, epoch))
            logging.info('Best Valid Hit@1  = {}({}) at epoch {}'.format(best_hit1_valid, best_hit1_valid_hit10, best_hit1_valid_epoch))
            logging.info('Best Valid Hit@10 = {}({}) at epoch {}'.format(best_hit10_valid, best_hit10_valid_hit1, best_hit10_valid_epoch))
            logging.info('Test @ Best Valid = {}({}), mrr={} at epoch {}'.format(record_hit1, record_hit10, record_mrr, record_epoch))
            logging.info('Best Test  Hit@1  = {}({}) at epoch {}'.format(best_hit1_test, best_hit1_test_hit10, best_hit1_test_epoch))
            logging.info('Best Test  Hit@10 = {}({}) at epoch {}'.format(best_hit10_test, best_hit10_test_hit1,best_hit10_test_epoch))
            logging.info("====================================")
