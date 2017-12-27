import numpy as np
import ranking
import cPickle
import scipy.sparse

def compute_performance(Xu_true, Xu_pred):
	NDCG = ranking.compute_ndcg(Xu_pred, Xu_true)
	MAP	= ranking.compute_map(Xu_pred, Xu_true)
	print "NDCG: ", NDCG, "MAP: ", MAP

filename = './data/Xu_train.pkl'
Xu_train = cPickle.load(open(filename, 'r'))
filename = './data/Xu_true.pkl'
Xu_true = cPickle.load(open(filename, 'r')).toarray()
filename = './data/Xu_test.pkl'
Xu_test = cPickle.load(open(filename, 'r')).toarray()
ntest_users, ntest_items = Xu_test.shape
filename = './data/Xs.pkl'
Xu = scipy.sparse.vstack((Xu_train, Xu_test))
U = cPickle.load(open('./data/U.pkl','r'))
V = cPickle.load(open('./data/V.pkl','r'))
# Xu_true = np.logical_xor(Xu_true.astype(int).astype(bool), Xu_test.astype(int).astype(bool)).astype(int)
Xu_true = Xu_true.astype(int)
Xu_predicted = np.dot(U[-ntest_users:, :], V.T)
Xu_predicted[Xu_predicted>0] = 1
Xu_predicted[Xu_predicted<0] = 0
Xu_predicted = Xu_predicted.astype(int)
print np.sum(np.logical_xor(Xu_true, Xu_predicted))/float(Xu_predicted.size)
# compute_performance(Xu_true, Xu_predicted)