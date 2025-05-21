import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from settings import *



class MyRawdataset(Dataset):
    def __init__(self, id_features_dict, id_adj_tensor_dict, att_features_dict, att_adj_tensor_dict, att_ids_dict, ent_ids_dict, is_neighbor=True):
        super(MyRawdataset, self).__init__()
        self.num = len(id_features_dict)  # number of samples

        self.nei = []
        self.nei_adj = None
        self.nei_id = []
        self.y = []

        self.att = []
        self.att_adj = None
        self.att_id = []

        for k in id_features_dict:
            if is_neighbor:
                if self.nei_adj==None:
                    self.nei_adj = id_adj_tensor_dict[k].unsqueeze(0)
                else:
                    self.nei_adj = torch.cat((self.nei_adj, id_adj_tensor_dict[k].unsqueeze(0)), dim=0)
                if self.att_adj==None:
                    self.att_adj = att_adj_tensor_dict[k].unsqueeze(0)
                else:
                    self.att_adj = torch.cat((self.att_adj, att_adj_tensor_dict[k].unsqueeze(0)), dim=0)


            self.nei.append(id_features_dict[k])
            self.nei_id.append(ent_ids_dict[k])
            self.y.append([k-1])
            self.att.append(att_features_dict[k])
            self.att_id.append(att_ids_dict[k])
            #self.y2_train.append([k])

        # transfer to tensor
        # if type(self.x_train[0]) is list:
        self.nei = torch.Tensor(self.nei)
        self.y = torch.Tensor(self.y).long()
        self.att = torch.Tensor(self.att)
        self.att_id = torch.Tensor(self.att_id).long()
        self.nei_id = torch.Tensor(self.nei_id).long()
        #self.y2_train = torch.Tensor(self.y2_train).long()



    # indexing
    def __getitem__(self, index):
        return self.nei[index], self.nei_id[index], self.nei_adj[index], self.y[index], self.att[index], self.att_id[index], self.att_adj[index]

    def __len__(self):
        return self.num


