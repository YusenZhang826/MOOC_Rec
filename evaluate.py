import torch
import math
import numpy as np

def Binary_CrossEntropy_loss(positive, negitive):
    positive_loss = -torch.log(torch.sigmoid(positive.squeeze()))
    negative_loss = -torch.sum(torch.log(1 - torch.sigmoid(negitive)), dim=1)
    loss = torch.sum(positive_loss + negative_loss)
    return loss


def BPR_loss(pos, negs):
    # pos = pos.squeeze(1)
    x = (pos) - (negs)
    loss = -torch.sum(torch.log(torch.sigmoid(x)), dim=1)
    loss = torch.mean(loss)
    return loss

def evaluation(preds, tgt_idx):
    sorted_idx = preds.argsort().tolist()[::-1]
    ranklist5 = sorted_idx[:5]
    ranklist10 = sorted_idx[:10]
    hr5 = getHitRatio(ranklist5, tgt_idx)
    hr10 = getHitRatio(ranklist10, tgt_idx)
    ndcg5 = getNDCG(ranklist5, tgt_idx)
    ndcg10 = getNDCG(ranklist10, tgt_idx)
    mrr5 = getMRR(ranklist5, tgt_idx)
    mrr10 = getMRR(ranklist10, tgt_idx)

    return hr5, ndcg5, hr10, ndcg10, mrr5, mrr10


def getHitRatio(ranklist, gtItem):
    for item in ranklist:
        if item == gtItem:
            return 1
    return 0

def getNDCG(ranklist, gtItem):
    for i in range(len(ranklist)):
        item = ranklist[i]
        if item == gtItem:
            return math.log(2) / math.log(i + 2)
    return 0

def getMRR(ranklist, gtItem):
    for index, item in enumerate(ranklist):
        if item == gtItem:
            return 1.0 / (index + 1.0)
    return 0