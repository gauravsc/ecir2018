import numpy as np
from sklearn.metrics import roc_auc_score


def compute_me(Xu_pred, Xu_true):
    Xu_true = Xu_true.astype(int)
    Xu_pred[Xu_pred>0] = 1
    Xu_pred[Xu_pred<0] = 0
    Xu_pred = Xu_pred.astype(int)
    ME = np.sum(np.logical_xor(Xu_true, Xu_pred))/float(Xu_pred.size)
    return ME

def compute_auc(Xu_pred, Xu_true):
    auc = []
    num_elements = np.sum(Xu_true, axis=1)
    Xu_true = Xu_true[num_elements>0, :]
    Xu_pred = Xu_pred[num_elements>0, :] 
    Xu_true = np.array(Xu_true)
    Xu_pred = np.array(Xu_pred)
    for i in xrange(Xu_true.shape[0]):
        auc.append(roc_auc_score(Xu_true[i,:], Xu_pred[i, :]))

    return np.mean(auc)



def dcg(relevances, rank=10):
    """Discounted cumulative gain at rank (DCG)"""
    relevances = np.asarray(relevances)[:rank]
    n_relevances = len(relevances)
    if n_relevances == 0:
        return 0.
    discounts = np.log2(np.arange(n_relevances) + 2)
    return np.sum(relevances / discounts)
 
 
def ndcg(relevances, rank=10):
    """Normalized discounted cumulative gain (NDGC)"""
    best_dcg = dcg(sorted(relevances, reverse=True), rank)
    if best_dcg == 0:
        return 0.
    return dcg(relevances, rank) / best_dcg


def compute_ndcg(Xu_pred, Xu_true):
    # Xu_true[Xu_true > 0] = 1
    relevances = []
    nDCG = 0
    for i in xrange(Xu_true.shape[0]):
        ind = np.argsort(Xu_pred[i, :])[::-1]
        # relevances.append(Xu_true[i, ind])
        relevances = Xu_true[i, ind]
        nDCG += ndcg(relevances, Xu_true.shape[1])

    return nDCG/Xu_true.shape[0]

def compute_map(Xu_pred, Xu_true):
    # Xu_true = Xu_true.toarray()
    Xu_true[Xu_true > 0] = 1
    relevances = []
    MAP = 0
    for i in xrange(Xu_true.shape[0]):
        ind = np.argsort(Xu_pred[i, :])[::-1]
        # ind = ind[0:20]
        # relevances.append(Xu_true[i, ind])
        relevances = Xu_true[i, ind]
        relevances = np.cumsum(relevances)
        # print relevances
        relevances = relevances/(range(len(relevances))+ np.ones(len(relevances)))
        MAP +=  (np.sum(relevances)/len(relevances))

    return MAP/Xu_true.shape[0]

