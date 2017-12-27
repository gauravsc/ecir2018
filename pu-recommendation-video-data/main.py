import numpy as np
import random as rd
import ranking
import cPickle
import scipy
import math
from scipy import linalg
from sklearn.preprocessing import normalize
import os


alpha = 1.0
dim = 100
lambd =0.0001
lambd_E = 1



def min_objective(Xu, Xs, W, last_U, last_V, t):
	nIterations = 1000000 + (t)* 500000
	learn_rate = 0.05
	nusers, nitems = Xu.shape
	_, nfeatures = Xs.shape
	nz_indices = np.nonzero(Xu)
	nz_elements = len(nz_indices[0])
	print "non zero entries are: ", nz_elements
	U = np.random.standard_normal((nusers, dim))
	U = normalize(U, norm='l2', axis=1)
	V = np.random.standard_normal((nitems, dim))
	V = normalize(V, norm='l2', axis=1)
	E = np.random.uniform(-1, 1, (nitems, dim))
	prev_V = np.array(V)
	prev_U = np.array(U)
	prev_E = np.array(E)
	skipped = 0
	# W = np.random.uniform(-0.001, 0.001, (dim, nfeatures))/100
	obj = 10
	for itera in xrange(nIterations):
		# print "iteration: ", itera
		if itera % 2 == 0:
			ind = rd.choice(xrange(nz_elements))
			i = nz_indices[0][ind]
			j = nz_indices[1][ind]
		else:
			i = rd.choice(xrange(nusers))
			j = rd.choice(xrange(nitems))
		# i = rd.choice(xrange(nusers))
		# j = rd.choice(xrange(nitems))
		if Xu[i, j] == 1:
			y = 1
		else:
			y = -1

		if last_U != None and y*np.dot(last_U[i,:], last_V[j, :]) < -.5:

			if itera > 1 and itera % 1000 == 0:
				obj =  np.sum((U-prev_U)**2) + np.sum((V-prev_V)**2) + np.sum((E-prev_E)**2)
				print "iterations: ", itera, " obj: ", obj
				prev_V = np.array(V)
				prev_U = np.array(U)
				prev_E = np.array(E)
				# learn_rate /= 1.0001
				if obj < 0.01:
					break
			skipped+=1		
			continue

		t = y*np.dot(U[i, :], V[j, :])

		if 1 -  t > 0:
			gradu = -1 * y * V[j, :]
			gradv = -1 * y * U[i, :]
		else:
			gradu = 0
			gradv = 0	
		
		U[i, :] = U[i, :] - learn_rate * (alpha * gradu) - learn_rate * lambd* 2 * U[i, :]
		# V[j, :] = V[j, :] - learn_rate * (alpha * gradv + (1-alpha) * (-2*np.dot(Xs[j, :].toarray(), W.T) + 2*np.dot((V[j,:]-E[j, :]), np.dot(W, W.T)))) - learn_rate * lambd * 2 * V[j, :]
		# E[j, :] = E[j, :] - learn_rate * ((1-alpha) * ( 2*np.dot(Xs[j, :].toarray(), W.T) - 2*np.dot((V[j,:]-E[j, :]), np.dot(W, W.T)))) - learn_rate * lambd_E * 2 * E[j, :]
		V[j, :] = V[j, :] - learn_rate * (alpha * gradv) - learn_rate * lambd * 2 * V[j, :]
		if  itera > 1 and itera % 1000 == 0:
			obj =  np.sum((U-prev_U)**2) + np.sum((V-prev_V)**2) + np.sum((E-prev_E)**2)
			print "iterations: ", itera, " obj: ", obj
			prev_V = np.array(V)
			prev_U = np.array(U)
			prev_E = np.array(E)
			# learn_rate /= 1.0001
			if obj < 0.01:
				break

		# W = W - learn_rate * ((1-alpha) *(-2*np.dot(V[j,:].reshape(1, -1).T, Xs[j, :].toarray().reshape(1, -1)) + 2* np.dot(np.dot(V[j,:].reshape(1, -1).T, V[j, :].reshape(1, -1)), W))) - learn_rate*lambd*2*W
		
		# if dump > 1:
		# 	print "before: ", Xu[i, j], dump
		# 	print "after: ", Xu[i, j], np.dot(U[i,:], V[j, :])
		# 	print grad1u, grad1v, grad2u, grad2v
		
		# if itera % 100 ==0:
		# 	su =0
		# 	for k in range(10000):
		# 		i = rd.choice(xrange(nusers))
		# 		j = rd.choice(xrange(nitems))
		# 		if Xu[i, j] == 0:
		# 			y = -1
		# 		else:
		# 			y = 1
		# 		if math.isnan(np.dot(U[i,:], V[j, :])):
		# 			print U[i,:], V[j, :], np.dot(U[i,:], V[j, :])
		# 			return 
		# 		su+= np.dot(y*U[i,:], V[j, :])
		# 	print "objective: ", su/10000.0
			# learn_rate /= 1.1
	print "skipped: ", skipped
	return U, V		

def train(Xu, Xs, W):
	last_U = None
	last_V = None
	times = 1
	for t in range(times):
		last_U, last_V = min_objective(Xu, Xs, W, last_U, last_V, t)
	return last_U, last_V

def obtain_W(Xs):
	nitems, nfeatures = Xs.shape
	V = np.random.uniform(-1, 1, (nitems, dim))
	W = np.random.uniform(-1, 1, (dim, nfeatures))
	delta = 0.1
	obj = 10
	i=0

	if os.path.isfile('./data/W.pkl'):
		W = cPickle.load(open('./data/W.pkl', 'r'))
		return W

	while obj > 1:
		i+=1
		prev_W = W
		prev_V = V
		W = linalg.solve(np.dot(V.T, V) + delta*np.identity(dim), Xs.T.dot(V).T)
		V = (linalg.solve(np.dot(W, W.T) + delta*np.identity(dim), Xs.dot(W.T).T)).T
		obj = np.sum((W-prev_W)**2) + np.sum((V-prev_V)**2)
		print "iteration: ", i,  "  objective: ", obj

	cPickle.dump(W, open('./data/W.pkl', 'w'))
	return W


def compute_performance(Xu_true, Xu_pred):
	NDCG = ranking.compute_ndcg(Xu_pred, Xu_true)
	MAP	= ranking.compute_map(Xu_pred, Xu_true)
	print "NDCG: ", NDCG, "MAP: ", MAP

import numpy as np
import random as rd
import ranking
import cPickle
import scipy
import math
from scipy import linalg
from sklearn.preprocessing import normalize
import os
from numpy.random import RandomState

alpha = 1.0
dim = 50
lambd =0.0001
lambd_E = 1
skipped_elements = None
nIterations = 10000000
# nIterations =3

def get_samples(Xu):
	nusers, nitems = Xu.shape
	samples = []
	nz_indices = np.nonzero(Xu)
	nz_elements = len(nz_indices[0])

	for itera in xrange(nIterations):
		if itera % 2 == 0:
			ind = rd.choice(xrange(nz_elements))
			i = nz_indices[0][ind]
			j = nz_indices[1][ind]
			samples.append((i, j))
		else:
			i = rd.choice(xrange(nusers))
			j = rd.choice(xrange(nitems))
			samples.append((i, j))
	return samples

def min_objective(Xu, Xs, W, last_U, last_V, samples):
	global skipped_elements
	learn_rate = 0.05
	nusers, nitems = Xu.shape
	_, nfeatures = Xs.shape
	nz_indices = np.nonzero(Xu)
	nz_elements = len(nz_indices[0])
	print "non zero entries are: ", nz_elements
	rs = RandomState(1234567890)
	U = rs.standard_normal((nusers, dim))
	U = normalize(U, norm='l2', axis=1)
	V = rs.standard_normal((nitems, dim))
	V = normalize(V, norm='l2', axis=1)
	E = np.random.standard_normal((nitems, dim))
	prev_V = np.array(V)
	prev_U = np.array(U)
	prev_E = np.array(E)
	skipped = 0
	skip_i = []
	skip_j = []
	# W = np.random.uniform(-0.001, 0.001, (dim, nfeatures))/100
	obj = 10
	for itera in xrange(nIterations):
		i = samples[itera][0]
		j = samples[itera][1]
		# print "iteration: ", itera
		# if itera % 2 == 0:
		# 	ind = rd.choice(xrange(nz_elements))
		# 	i = nz_indices[0][ind]
		# 	j = nz_indices[1][ind]
		# else:
		# 	i = rd.choice(xrange(nusers))
		# 	j = rd.choice(xrange(nitems))
		# i = rd.choice(xrange(nusers))
		# j = rd.choice(xrange(nitems))
		if Xu[i, j] > 0:
			y = 1
		else:
			y = -1

		if skipped_elements[i, j] == 1 or (last_U != None and y*np.dot(last_U[i,:], last_V[j, :]) < -0.5):
			if itera > 1 and itera % 50000 == 0:
				obj =  np.sum((U-prev_U)**2) + np.sum((V-prev_V)**2) + np.sum((E-prev_E)**2)
				print "iterations: ", itera, " obj: ", obj
				prev_V = np.array(V)
				prev_U = np.array(U)
				prev_E = np.array(E)
				# learn_rate /= 1.0001
				if obj < 0.01:
					break
			skipped+=1
			skip_i.append(i)
			skip_j.append(j)		
			continue
			# skipped_elements[i, j] = 1
			

		t = y*np.dot(U[i, :], V[j, :])

		if 1 -  t > 0:
			gradu = -1 * y * V[j, :]
			gradv = -1 * y * U[i, :]
		else:
			gradu = 0
			gradv = 0	
		
		# print "original: ", y
		# print "before: ",  np.dot(U[i, :], V[j, :])

		U[i, :] = U[i, :] - learn_rate * (alpha * gradu) - learn_rate * lambd* 2 * U[i, :]
		# V[j, :] = V[j, :] - learn_rate * (alpha * gradv + (1-alpha) * (-2*np.dot(Xs[j, :].toarray(), W.T) + 2*np.dot((V[j,:]-E[j, :]), np.dot(W, W.T)))) - learn_rate * lambd * 2 * V[j, :]
		# E[j, :] = E[j, :] - learn_rate * ((1-alpha) * ( 2*np.dot(Xs[j, :].toarray(), W.T) - 2*np.dot((V[j,:]-E[j, :]), np.dot(W, W.T)))) - learn_rate * lambd_E * 2 * E[j, :]
		V[j, :] = V[j, :] - learn_rate * (alpha * gradv) - learn_rate * lambd * 2 * V[j, :]
		
		# print "after: ", np.dot(U[i, :], V[j, :])
		if  itera > 1 and itera % 50000 == 0:
			obj =  np.sum((U-prev_U)**2) + np.sum((V-prev_V)**2) + np.sum((E-prev_E)**2)
			print "iterations: ", itera, " obj: ", obj
			prev_V = np.array(V)
			prev_U = np.array(U)
			prev_E = np.array(E)
			# learn_rate /= 1.0001
			if obj < 0.01:
				break

		# W = W - learn_rate * ((1-alpha) *(-2*np.dot(V[j,:].reshape(1, -1).T, Xs[j, :].toarray().reshape(1, -1)) + 2* np.dot(np.dot(V[j,:].reshape(1, -1).T, V[j, :].reshape(1, -1)), W))) - learn_rate*lambd*2*W
		
		# if dump > 1:
		# 	print "before: ", Xu[i, j], dump
		# 	print "after: ", Xu[i, j], np.dot(U[i,:], V[j, :])
		# 	print grad1u, grad1v, grad2u, grad2v
		
		# if itera % 1000 ==0:
		# 	su =0
		# 	for k in range(10000):
		# 		i = rd.choice(xrange(nusers))
		# 		j = rd.choice(xrange(nitems))
		# 		if Xu[i, j] == 0:
		# 			y = -1
		# 		else:
		# 			y = 1
		# 		if math.isnan(np.dot(U[i,:], V[j, :])):
		# 			print U[i,:], V[j, :], np.dot(U[i,:], V[j, :])
		# 			return 
		# 		su+= np.dot(y*U[i,:], V[j, :])
		# 	print "objective: ", su/10000.0
		# 	# learn_rate /= 1.1
	print "skipped: ", skipped
	skipped_elements[skip_i, skip_j] = 1
	return U, V		

def train(Xu, Xs, W, samples):
	last_U = None
	last_V = None
	times = 3
	# for t in range(times):
	# 	last_U, last_V = min_objective(Xu, Xs, W, last_U, last_V, t)
	diff_non_zero = 1000
	ctr = 0
	while diff_non_zero > 100 or len(skipped_elements.nonzero()[0]) == 0:
		prev_skipped = len(skipped_elements.nonzero()[0])
		last_U, last_V = min_objective(Xu, Xs, W, last_U, last_V, samples)
		curr_skipped = len(skipped_elements.nonzero()[0])
		diff_non_zero = curr_skipped - prev_skipped
		print "New elements added to skip list: ", diff_non_zero
		ctr += 1
		if ctr == 6:
			break
	return last_U, last_V

def obtain_W(Xs):
	nitems, nfeatures = Xs.shape
	V = np.random.uniform(-1, 1, (nitems, dim))
	W = np.random.uniform(-1, 1, (dim, nfeatures))
	delta = 0.1
	obj = 10
	i=0

	if os.path.isfile('./data/W.pkl'):
		W = cPickle.load(open('./data/W.pkl', 'r'))
		return W

	while obj > 1:
		i+=1
		prev_W = W
		prev_V = V
		W = linalg.solve(np.dot(V.T, V) + delta*np.identity(dim), Xs.T.dot(V).T)
		V = (linalg.solve(np.dot(W, W.T) + delta*np.identity(dim), Xs.dot(W.T).T)).T
		obj = np.sum((W-prev_W)**2) + np.sum((V-prev_V)**2)
		print "iteration: ", i,  "  objective: ", obj

	cPickle.dump(W, open('./data/W.pkl', 'w'))
	return W


def compute_performance(Xu_true, Xu_pred):
	NDCG = ranking.compute_ndcg(Xu_pred, Xu_true)
	MAP	= ranking.compute_map(Xu_pred, Xu_true)
	ME = ranking.compute_me(Xu_pred, Xu_true)
	AUC = ranking.compute_auc(Xu_pred, Xu_true)
	print "NDCG: ", NDCG, "MAP: ", MAP, "MISCLASSIFICATION ERROR: ", ME, "AUC: ", AUC


def main():
	global skipped_elements
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
	W = obtain_W(Xs)
	Xu = scipy.sparse.vstack((Xu_train, Xu_test))
	skipped_elements = scipy.sparse.csr_matrix(Xu.shape)
	samples = get_samples(Xu)
	U, V = train(Xu, Xs, W, samples)
	cPickle.dump(U, open('./data/U.pkl','w'))
	cPickle.dump(V, open('./data/V.pkl','w'))
	Xu_true = np.logical_xor(Xu_true.astype(int).astype(bool), Xu_test.toarray().astype(int).astype(bool)).astype(int)
	Xu_predicted = np.dot(U[-ntest_users:, :], V.T)
	compute_performance(Xu_true, Xu_predicted)

main()