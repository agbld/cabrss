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

class XbmSoftMarginTripletLoss(nn.Module):
    def __init__(self, model: SentenceTransformer, distance_metric=TripletDistanceMetric.EUCLIDEAN):
        super(XbmSoftMarginTripletLoss, self).__init__()
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

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor, epoch):

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

        distance_pos = self.distance_metric(rep_anchor, rep_pos)
        distance_neg = self.distance_metric(rep_anchor, rep_neg)

        # losses = F.relu(distance_pos - distance_neg + self.triplet_margin)
        tl = torch.log1p(torch.exp(distance_pos - distance_neg))
        soft_margin_triplet_loss = tl.mean()

        self.xbm.enqueue_dequeue(current_anchor.detach(), current_pos.detach(), current_neg.detach())

        return soft_margin_triplet_loss