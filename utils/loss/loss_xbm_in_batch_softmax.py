import torch
from torch import nn, Tensor
from typing import Union, Tuple, List, Iterable, Dict
import torch.nn.functional as F
from enum import Enum
from sentence_transformers import SentenceTransformer
from utils.xbm import triplet_XBM
import config

class TripletDistanceMetric(Enum):
    """
    The metric for the triplet loss
    """
    COSINE = lambda x, y: 1 - F.cosine_similarity(x, y)
    EUCLIDEAN = lambda x, y: F.pairwise_distance(x, y, p=2)
    MANHATTAN = lambda x, y: F.pairwise_distance(x, y, p=1)

class XbmBatchSoftmaxLoss(nn.Module):

    def __init__(self, model: SentenceTransformer, distance_metric=TripletDistanceMetric.EUCLIDEAN):
        super(XbmBatchSoftmaxLoss, self).__init__()
        self.model = model
        self.distance_metric = distance_metric
        self.xbm = triplet_XBM(config.xbm_size)


    def get_config_dict(self):
        distance_metric_name = self.distance_metric.__name__
        for name, value in vars(TripletDistanceMetric).items():
            if value == self.distance_metric:
                distance_metric_name = "TripletDistanceMetric.{}".format(name)
                break

        return {'distance_metric': distance_metric_name}

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], label: Tensor, epoch):
        reps = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
        current_anchor, current_pos, current_neg = reps

        # Cross-Batch Memory (xbm)
        if epoch >= config.xbm_start_iteration:
            xbm_anchor, xbm_pos, xbm_neg = self.xbm.get()
            rep_anchor = torch.cat([current_anchor, xbm_anchor])
            rep_pos = torch.cat([current_pos, xbm_pos])
            rep_neg = torch.cat([current_neg, xbm_neg])
        else:
            rep_anchor = current_anchor
            rep_pos = current_pos
            rep_neg = current_neg

        loss_in_batch = torch.tensor([]).to(config.device)
        for i, i_rep in enumerate(rep_anchor):
            # 計算 anchor 跟 in-batch 負樣本（其他人的正樣本）的距離
            distance_neg = 50 - self.distance_metric(i_rep, rep_pos)

            # 計算 logsoftmax
            logsoftmax = nn.LogSoftmax(dim=1)
            output = logsoftmax(distance_neg.unsqueeze(dim=0)) * (-1)

            # 取出正樣本對應的 logsoftmax loss
            loss = output[0][i].unsqueeze(dim=0)
            
            loss_in_batch = torch.cat((loss_in_batch, loss), 0)

        self.xbm.enqueue_dequeue(current_anchor.detach(), current_pos.detach(), current_neg.detach())

        return loss_in_batch.mean()