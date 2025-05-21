import os
from os.path import abspath, dirname, join
import torch
import torch.nn as nn
from torch.nn import *
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import torch.utils.data as Data
from loader.DBP15KRawNeighbors import DBP15KRawNeighbors
import random
import pandas as pd
import argparse
import logging
from datetime import datetime
from settings import *

from model.GATs_layers import *
from model.simCLR_run import *

def parse_options(parser):
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--time', type=str, default=datetime.now().strftime("%Y%m%d%H%M%S"))
    parser.add_argument('--dataset', type=str, default='DBP15K')
    parser.add_argument('--language', type=str, default='zh_en')
    parser.add_argument('--model', type=str, default='LaBSE')
    parser.add_argument('--seed', type=int, default=37)
    parser.add_argument('--epoch', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--center_norm', type=bool, default=False)
    parser.add_argument('--neighbor_norm', type=bool, default=True)
    parser.add_argument('--emb_norm', type=bool, default=True)
    parser.add_argument('--combine', type=bool, default=True)
    parser.add_argument('--momentum', type=float, default=0.9999)
    parser.add_argument('--gat_num', type=int, default=1)
    parser.add_argument('--loss_choice', type=str, default='gcl')
    parser.add_argument('--t', type=float, default=0.08)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--gda', action='store_true', default=True, help='to use graph data augment')
    parser.add_argument('--att', action='store_true', default=True, help='to use attribute')
    parser.add_argument('--hard', action='store_true', default=True, help='to use hard negative mining')
    parser.add_argument('--drop_rate', type=float, default=0.2)

    parser.add_argument('--threshold', default=0.9, type=float, help='method')
    parser.add_argument('--cos', action='store_true', default=True, help='use cosine lr schedule')
    parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,help='learning rate schedule (when to drop lr by 10x)')
    parser.add_argument('--epoch_start', type=int, default=80)
    parser.add_argument('--weight_init', type=float, default=0.05)
    parser.add_argument('--beta', type=float, default=0.1)
    parser.add_argument('--tau_plus', type=float, default=0.0)
    parser.add_argument('--estimator', default='hard', type=str, help='Choose hclloss function')

    return parser.parse_args()



if __name__ == "__main__":
    # prepare
    parser = argparse.ArgumentParser()
    args = parse_options(parser)
    Mylogging(args)
    logging.info(args)
    start_loop = main_loop(args)
    start_loop.train()






