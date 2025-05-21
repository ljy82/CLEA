from loader.DWY100KRawNeighbors import DBP100KRawNeighbors, DBP100KattNeighbors
from loader.deal_raw_dataset import *
import torch.utils.data as Data
import pandas as pd
from settings import *

class dwy100kdataset():
    def __init__(self, args):
        self.args = args
        loader1 = DBP100KRawNeighbors(self.args.language, "1")
        attrloader1 = DBP100KattNeighbors(self.args.language, "1")
        self.ent_reinfo1 = loader1.ent_reinfo
        self.attr_reinfo1 = attrloader1.attr_reinfo
        myset1 = MyRawdataset(loader1.id_neighbors_dict, loader1.id_adj_tensor_dict, attrloader1.id_neighbors_dict, attrloader1.id_adj_tensor_dict, attrloader1.id_attr_dict, loader1.id_ent_dict)
        del loader1, attrloader1



        self.loader1 = Data.DataLoader(
            dataset=myset1,  # torch TensorDataset format
            batch_size=self.args.batch_size,  # all test data
            shuffle=True,
            drop_last=False,
        )

        self.eval_loader1 = Data.DataLoader(
            dataset=myset1,  # torch TensorDataset format
            batch_size=self.args.batch_size,  # all test data
            shuffle=True,
            drop_last=False,
        )
        del myset1

        loader2 = DBP100KRawNeighbors(self.args.language, "2")
        attrloader2 = DBP100KattNeighbors(self.args.language, "2")
        self.ent_reinfo2 = loader2.ent_reinfo
        self.attr_reinfo2 = attrloader2.attr_reinfo
        myset2 = MyRawdataset(loader2.id_neighbors_dict, loader2.id_adj_tensor_dict, attrloader2.id_neighbors_dict, attrloader2.id_adj_tensor_dict, attrloader2.id_attr_dict, loader2.id_ent_dict)
        del loader2, attrloader2

        self.loader2 = Data.DataLoader(
            dataset=myset2,  # torch TensorDataset format
            batch_size=self.args.batch_size,  # all test data
            shuffle=True,
            drop_last=False,
        )

        self.eval_loader2 = Data.DataLoader(
            dataset=myset2,  # torch TensorDataset format
            batch_size=self.args.batch_size,  # all test data
            shuffle=True,
            drop_last=False,
        )
        del myset2


    # get the linked entity ids
        def link_loader(mode, valid=False):
            link = {}
            if valid == False:
                f = 'test.ref'
            else:
                f = 'valid.ref'
            link_data = pd.read_csv(join(join(DATA_DIR, 'DWY100K', mode), f), sep='\t', header=None)
            link_data.columns = ['entity1', 'entity2']
            entity1_id = link_data['entity1'].values.tolist()
            entity2_id = link_data['entity2'].values.tolist()
            for i, _ in enumerate(entity1_id):
                link[entity1_id[i]] = entity2_id[i]
                link[entity2_id[i]] = entity1_id[i]
            return link

        self.test_link = link_loader(self.args.language)
        self.val_link = link_loader(self.args.language, True)

    def getdata(self):
        return self.loader1, self.loader2, self.eval_loader1, self.eval_loader2, self.val_link, self.test_link, self.ent_reinfo1, self.ent_reinfo2, self.attr_reinfo1, self.attr_reinfo2


