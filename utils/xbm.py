# reference: Cross-Batch Memory for Embedding Learning - https://arxiv.org/pdf/1912.06798.pdf
import torch

class triplet_XBM:
    def __init__(self, K):
        self.K = K
        self.anchor = torch.tensor([]).cuda()
        self.pos = torch.tensor([]).cuda()
        self.neg = torch.tensor([]).cuda()

    def get(self):
            return self.anchor, self.pos, self.neg

    def enqueue_dequeue(self, anchor, pos, neg):
        self.anchor = torch.cat((self.anchor, anchor.cuda()), 0)[-self.K:]
        self.pos = torch.cat((self.pos, pos.cuda()), 0)[-self.K:]
        self.neg = torch.cat((self.neg, neg.cuda()), 0)[-self.K:]