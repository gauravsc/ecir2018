import numpy as np
import random as rd
import ranking
import cPickle
import scipy
import math
from scipy import linalg
from sklearn.preprocessing import normalize
import os
import math


alpha = 1.0
dim = 50
lambd =0.0001
lambd_E = 1
nIterations = 10000000

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def compute_objective(Xu, U, V):
	nusers, nitems = Xu.shape
	nz_elements = Xu.nonzero()
	cnt = 0
	for l in xrange(10000):
		t = rd.choice(xrange(len(nz_elements[0])))
		u = nz_elements[0][t]
		i = nz_elements[1][t]
		j = rd.choice(xrange(nitems))
		if Xu[u, i] > Xu[u, j] and np.dot(U[u, :], V[i, :]-V[j, :]) > 0:
			cnt += 1
		elif Xu[u, i] < Xu[u, j] and np.dot(U[u, :], V[i, :]-V[j, :]) < 0:
			cnt += 1
	return 10000-cnt

def train(Xu):
	learn_rate = 0.05
	nusers, nitems = Xu.shape
	U = np.random.standard_normal((nusers, dim))
	V = np.random.standard_normal((nitems, dim))
	U = normalize(U, norm='l2', axis=1)
	V = normalize(V, norm='l2', axis=1)
	nz_elements = Xu.nonzero()
	for l in xrange(nIterations):
		t = rd.choice(xrange(len(nz_elements[0])))
		u = nz_elements[0][t]
		i = nz_elements[1][t]
		j = rd.choice(xrange(nitems))
		while Xu[u, j] == 1:
			j = rd.choice(xrange(nitems))

		U[u, :] = U[u, :] + learn_rate * (1 - sigmoid(np.dot(U[u, :], V[i, :]-V[j, :]))) * (V[i, :]-V[j, :])
		V[i, :] = V[i, :] + learn_rate * (1 - sigmoid(np.dot(U[u, :], V[i, :]-V[j, :]))) * U[u, :]
		V[j, :]	= V[j, :] + learn_rate * (1 - sigmoid(np.dot(U[u, :], V[i, :]-V[j, :]))) * -U[u, :]

		if l % 10000 == 0:
			learn_rate /= 1.0001
			print "Iterations: ", l, "  Objective: ", compute_objective(Xu, U, V)

	return U, V		



def compute_performance(Xu_true, Xu_pred):
	NDCG = ranking.compute_ndcg(Xu_pred, Xu_true)
	MAP	= ranking.compute_map(Xu_pred, Xu_true)
	ME = ranking.compute_me(Xu_pred, Xu_true)
	AUC = ranking.compute_auc(Xu_pred, Xu_true)
	print "NDCG: ", NDCG, "MAP: ", MAP, "MISCLASSIFICATION ERROR: ", ME, "AUC: ", AUC


def main():
	filename = './data/Xu_train.pkl'
	Xu_train = cPickle.load(open(filename, 'r'))
	filename = './data/Xu_true.pkl'
	Xu_true = cPickle.load(open(filename, 'r')).toarray()
	filename = './data/Xu_test.pkl'
	Xu_test = cPickle.load(open(filename, 'r'))
	ntest_users, ntest_items = Xu_test.shape
	filename = './data/Xs.pkl'
	Xs = cPickle.load(open(filename, 'r'))
	Xs = normalize(Xs, norm='l1', axis=1)
	Xu = scipy.sparse.vstack((Xu_train, Xu_test))
	skipped_elements = scipy.sparse.csr_matrix(Xu.shape)
	U, V = train(Xu)
	cPickle.dump(U, open('./data/U.pkl','w'))
	cPickle.dump(V, open('./data/V.pkl','w'))
	Xu_true = np.logical_xor(Xu_true.astype(int).astype(bool), Xu_test.toarray().astype(int).astype(bool)).astype(int)
	Xu_predicted = np.dot(U[-ntest_users:, :], V.T)
	compute_performance(Xu_true, Xu_predicted)

main()