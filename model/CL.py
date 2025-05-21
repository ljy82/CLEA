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

class DynamicSpectralClustering:
    def __init__(self, n_initial_clusters=100, merge_thresh=0.5):
        self.n_clusters = n_initial_clusters
        self.merge_thresh = merge_thresh
    
    def incremental_update(self, embeddings, prev_clusters, llm_categories):
        # 计算带类别权重的相似度矩阵
        S = pairwise_distances(embeddings, metric='euclidean')
        S += beta * jaccard_similarity(llm_categories)
        
        # Nyström近似与增量特征分解
        sampled_indices = np.random.choice(len(embeddings), int(0.2*len(embeddings)))
        S_sampled = S[sampled_indices][:, sampled_indices]
        eigenvectors = nystroem_approx(S_sampled)
        
        # 更新簇分配并合并/分裂
        new_labels = kmeans(eigenvectors, self.n_clusters)
        new_clusters = adjust_clusters(new_labels, prev_clusters, self.merge_thresh)
        
        return new_clusters

# 训练循环中调用
for epoch in range(total_epochs):
    train_one_epoch(model)
    if epoch % T == 0:
        embeddings = model.get_embeddings()
        clusters = clusterer.incremental_update(embeddings, clusters, llm_categories)






