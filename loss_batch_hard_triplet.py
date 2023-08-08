import torch
from torch import nn, Tensor
from typing import Union, Tuple, List, Iterable, Dict
import torch.nn.functional as F
from enum import Enum
from sentence_transformers import SentenceTransformer
import config

class TripletDistanceMetric(Enum):
    """
    The metric for the triplet loss
    """
    COSINE = lambda x, y: 1 - F.cosine_similarity(x, y)
    EUCLIDEAN = lambda x, y: F.pairwise_distance(x, y, p=2)
    MANHATTAN = lambda x, y: F.pairwise_distance(x, y, p=1)

class BatchHardTripletLoss(nn.Module):

    def __init__(self, model: SentenceTransformer, distance_metric=TripletDistanceMetric.EUCLIDEAN):
        super(BatchHardTripletLoss, self).__init__()
        self.model = model
        self.distance_metric = distance_metric


    def get_config_dict(self):
        distance_metric_name = self.distance_metric.__name__
        for name, value in vars(TripletDistanceMetric).items():
            if value == self.distance_metric:
                distance_metric_name = "TripletDistanceMetric.{}".format(name)
                break

        return {'distance_metric': distance_metric_name}

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], label: Tensor, epoch):
        reps = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]

        rep_anchor, rep_pos, rep_neg = reps
        distance_pos = self.distance_metric(rep_anchor, rep_pos)

        loss_in_batch = torch.tensor([]).to(config.device)
        for i, i_rep in enumerate(rep_anchor):
            # anchor as negative
            anc_as_neg = torch.cat([rep_anchor[0:i], rep_anchor[i+1:]]) # Skip the current anchor itself
            # other pos as negative
            pos_as_neg = torch.cat([rep_pos[0:i], rep_pos[i+1:]]) # Skip the current pos itself
            # other neg as negative
            other_neg_as_neg = rep_neg

            neg_vector = torch.cat([anc_as_neg, pos_as_neg, other_neg_as_neg]) 
            distance_neg = self.distance_metric(i_rep, neg_vector)

            # find the hardest neg in a mini-batch
            hardest_neg = min(distance_neg)
            tl = torch.log1p(torch.exp(distance_pos[i] - hardest_neg))
            loss_in_batch = torch.cat((loss_in_batch, tl.unsqueeze(dim=0)), 0)

        return loss_in_batch.mean()
