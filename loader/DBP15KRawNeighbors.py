import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from settings import *
import csv
import pandas as pd
import torch
import pickle
from collections import OrderedDict
def standardize(lst):
    mean = np.mean(lst)
    std = np.std(lst)
    standardized_lst = [x / std for x in lst]
    return standardized_lst

def normalize(lst):
    min_value = min(lst)
    max_value = max(lst)
    normalized_lst = [(x - min_value) / (max_value - min_value) for x in lst]
    return normalized_lst

class DBP15KRawNeighbors():
    def __init__(self, language, doc_id):
        self.language = language
        self.doc_id = doc_id
        self.path = join(DATA_DIR, 'DBP15K', self.language)
        self.id_entity = {}
        # self.id_neighbor_loader = {}
        self.id_adj_tensor_dict = {}
        self.id_neighbors_dict = {}
        self.id_ent_dict = {}
        self.ent_reinfo = []
        self.load()
        self.id_neighbors_loader()
        self.get_center_adj()

    # raw_LaBSE_emb_
    def load(self):
        with open(join(self.path, "raw_LaBSE_emb_" + self.doc_id + '.pkl'), 'rb') as f:
            self.id_entity = pickle.load(f)

       


    def id_neighbors_loader(self):
        data = pd.read_csv(join(self.path, 'triples_' + self.doc_id), header=None, sep='\t')

        data.columns = ['head', 'relation', 'tail']
        #data.sort_values(by=['head'], inplace=True)
        # self.id_neighbor_loader = {head: {relation: [neighbor1, neighbor2, ...]}}
        size = max(self.id_entity.keys())+2
        self.ent_reinfo = [0 for _ in range(size)]
        self.ent_reinfo[0] = 999
        for index, row in data.iterrows():
            # head-rel-tail, tail is a neighbor of head
            # print("int(row['head']): ", int(row['head']))
            head_str = self.id_entity[int(row['head'])][0]
            tail_str = self.id_entity[int(row['tail'])][0]
            head_id = int(row['head']) + 1
            tail_id = int(row['tail']) + 1

            self.ent_reinfo[head_id] = self.ent_reinfo[head_id] + 1
            self.ent_reinfo[tail_id] = self.ent_reinfo[tail_id] + 1

            if not head_id in self.id_neighbors_dict.keys():
                self.id_neighbors_dict[head_id] = [head_str]
                self.id_ent_dict[head_id] = [head_id]
            if not tail_str in self.id_neighbors_dict[head_id]:
                self.id_neighbors_dict[head_id].append(tail_str)
                self.id_ent_dict[head_id].append(tail_id)
            
            if not tail_id in self.id_neighbors_dict.keys():
                self.id_neighbors_dict[tail_id] = [tail_str]
                self.id_ent_dict[tail_id] = [tail_id]
            if not head_str in self.id_neighbors_dict[tail_id]:
                self.id_neighbors_dict[tail_id].append(head_str)
                self.id_ent_dict[tail_id].append(head_id)

        
        self.ent_reinfo = torch.Tensor(self.ent_reinfo)
        self.ent_reinfo = torch.log(self.ent_reinfo)
        self.ent_reinfo = (self.ent_reinfo.max() - self.ent_reinfo) / (self.ent_reinfo.max() - self.ent_reinfo.mean())
        
    
    def get_adj(self, valid_len):
        adj = torch.zeros(NEIGHBOR_SIZE, NEIGHBOR_SIZE).bool()
        for i in range(0, valid_len):
            adj[i, i] = 1
            adj[0, i] = 1
            adj[i, 0] = 1
        return adj

    def get_center_adj(self):
        for k, v in self.id_neighbors_dict.items():

            if len(v) < NEIGHBOR_SIZE:
                self.id_adj_tensor_dict[k] = self.get_adj(len(v))
                self.id_neighbors_dict[k] = v + [[0]*LaBSE_DIM] * (NEIGHBOR_SIZE - len(v))
                self.id_ent_dict[k] = self.id_ent_dict[k] + [0] * (NEIGHBOR_SIZE - len(self.id_ent_dict[k]))
            else:
                self.id_adj_tensor_dict[k] = self.get_adj(NEIGHBOR_SIZE)
                self.id_neighbors_dict[k] = v[:NEIGHBOR_SIZE]
                self.id_ent_dict[k] = self.id_ent_dict[k][:NEIGHBOR_SIZE]


class DBP15KattNeighbors():
    def __init__(self, language, doc_id):
        self.language = language
        self.doc_id = doc_id
        self.path = join(DATA_DIR, 'DBP15K', self.language)
        self.id_entity = {}
        self.attr_reinfo = {}
        # self.id_neighbor_loader = {}
        self.id_adj_tensor_dict = {}
        self.id_neighbors_dict = {}
        self.id_attr_dict = {}
        self.load()
        self.att_load()
        self.id_neighbors_loader()
        self.get_center_adj()


    # raw_LaBSE_emb_
    def load(self):
        with open(join(self.path, "raw_LaBSE_emb_" + self.doc_id + '.pkl'), 'rb') as f:
            self.id_entity = pickle.load(f)

        #self.attr = load_dict(join(self.path, "attr_id_" + self.doc_id))

        

    def att_load(self):
        with open(join(self.path, "attr_" + self.doc_id + '.pkl'), 'rb') as f:
            self.id_attr = pickle.load(f)

            


    def id_neighbors_loader(self):
        data = pd.read_csv(join(self.path, 'att_triples_' + self.doc_id), header=None, sep='\t')

        data.columns = ['head', 'tail', 'value']
        self.attr_reinfo_tmp = load_dict(join(self.path, 'attr_ids_' + self.doc_id))
        for k,v in self.attr_reinfo_tmp.items():
            k = int(k)+1
            self.attr_reinfo[k] = 1
        self.attr_reinfo[0] = 9999

        for index, row in data.iterrows():

            head_str = self.id_entity[int(row['head'])][0]
            tail_str = self.id_attr[int(row['tail'])][0]
            head_id = int(row['head']) + 1
            tail_id = int(row['tail']) + 1

            self.attr_reinfo[tail_id] = self.attr_reinfo[tail_id] + 1


            if not head_id in self.id_neighbors_dict.keys():
                self.id_neighbors_dict[head_id] = [self.id_entity[head_id-1][0]]
                self.id_attr_dict[head_id] = [head_id]
            if not tail_str in self.id_neighbors_dict[head_id]:
                self.id_neighbors_dict[head_id].append(tail_str)
                self.id_attr_dict[head_id].append(tail_id)


        self.attr_reinfo = dict(sorted(self.attr_reinfo.items())) #sorted by key
        tmp = sorted(self.attr_reinfo.items())
        self.attr_reinfo = [self.attr_reinfo[key] for key in self.attr_reinfo]
        self.attr_reinfo = torch.Tensor(self.attr_reinfo)
        self.attr_reinfo = torch.log(self.attr_reinfo)
        self.attr_reinfo = (self.attr_reinfo.max() - self.attr_reinfo) / (self.attr_reinfo.max() - self.attr_reinfo.mean())

        for key in self.id_entity:
            key = key + 1
            if not key in self.id_neighbors_dict.keys():
                self.id_neighbors_dict[key] = [self.id_entity[key-1][0]]
                self.id_attr_dict[key] = [key]


    def get_adj(self, valid_len):
        adj = torch.zeros(ATT_NEIGHBOR_SIZE, ATT_NEIGHBOR_SIZE).bool()
        for i in range(0, valid_len):
            adj[i, i] = 1
            adj[0, i] = 1
            adj[i, 0] = 1
        return adj

    def get_center_adj(self):

        for k, v in self.id_neighbors_dict.items():
            if len(v) < ATT_NEIGHBOR_SIZE:
                self.id_adj_tensor_dict[k] = self.get_adj(len(v))
                self.id_neighbors_dict[k] = v + [[0] * LaBSE_DIM] * (ATT_NEIGHBOR_SIZE - len(v))
                self.id_attr_dict[k] = self.id_attr_dict[k] + [0] * (ATT_NEIGHBOR_SIZE - len(self.id_attr_dict[k]))

            else:
                self.id_adj_tensor_dict[k] = self.get_adj(ATT_NEIGHBOR_SIZE)
                self.id_neighbors_dict[k] = v[:ATT_NEIGHBOR_SIZE]
                self.id_attr_dict[k] = self.id_attr_dict[k][:ATT_NEIGHBOR_SIZE]





def load_dict(file_path, read_kv='kv', sep='\t'):
    '''
    :param file_path:
    :param read_kv: ='kv' or 'vk'
    :return:
    '''
    print("load dict:", file_path)
    value_trans_dict = {}
    with open(file_path, encoding='utf-8', mode='r') as f:
        if read_kv == 'kv':
            for line in f:
                th = line[:-1].split(sep)
                if len(th) == 2:
                    value_trans_dict[th[0]] = th[1]  # id:value
                else:
                    value_trans_dict[th[0]] = ' '.join(th[1:])
        else:  # vk
            for line in f:
                th = line[:-1].split(sep)
                value_trans_dict[th[1]] = th[0]  # value:id

    return value_trans_dict