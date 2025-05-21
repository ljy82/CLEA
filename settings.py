import os
from os.path import abspath, dirname, join, exists
from collections import defaultdict
import json
import codecs
import csv
from tqdm import tqdm
import pickle
import random
import numpy as np
import torch
import logging
from datetime import datetime

def fix_seed(seed=37):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.


PROJ_DIR = abspath(dirname(__file__))
LINK_DIR = join(PROJ_DIR, 'link')
CLIENT_DIR = join(PROJ_DIR, 'client')
DATA_DIR = join(PROJ_DIR, 'data/')
RAW_DATA_DIR = join(DATA_DIR, 'raw_data')
FUZZY_DIR = join(DATA_DIR, 'fuzzy')
CANDIDATE_DIR = join(PROJ_DIR, 'candidates')
os.makedirs(DATA_DIR, exist_ok=True)
EVAL_DIR = join(PROJ_DIR, 'evaluate')


TOKEN_LEN = 50
VOCAB_SIZE = 100000
LaBSE_DIM = 768
EMBED_DIM = 300
BATCH_SIZE = 96
FASTTEXT_DIM = 300
NEIGHBOR_SIZE = 30
ATT_NEIGHBOR_SIZE = 30
ATTENTION_DIM = 300
MULTI_HEAD_DIM = 1

LINK_LEN = 15000

# directory for datasets
EXPAND_DIR = join(DATA_DIR, 'expand')

# split proportion
train_prop = 1
test_prop = 1 - train_prop


def Mylogging(args):
    if not os.path.exists(join(PROJ_DIR, 'log')):
        os.mkdir(join(PROJ_DIR, 'log'))
    logging.basicConfig(filename=join(PROJ_DIR, 'log', '{}_{}_{}_{}_{}_{}_{}_{}.log'.format(
        datetime.now().strftime("%Y%m%d%H%M%S"),
        args.language,
        args.batch_size,
        args.att,
        args.gda,
        args.t,
        args.momentum,
        args.lr
    )), level=logging.INFO,datefmt='## %Y-%m-%d %H:%M:%S',format='%(asctime)s.%(msecs)03d [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s',filemode = 'a')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('LINE %(lineno)-4d : %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

