import cPickle
import random as rd
import scipy.sparse 
import numpy as np
rd.seed(3)
num_trains = 100000
num_test = 7000
filename = './data/Xu.pkl'
Xu = cPickle.load(open(filename, 'r'))
train_Xu = Xu[0:num_trains, :]
cPickle.dump(train_Xu, open('./data/Xu_train.pkl','w'))
gtruth_Xu = Xu[num_trains: num_trains+num_test, :]
indices = scipy.nonzero(gtruth_Xu)
k = rd.sample(range(len(indices[0])), int(0.5*len(indices[0])))
gtest_Xu = scipy.sparse.csr_matrix(gtruth_Xu, copy=True)
gtest_Xu[indices[0][k], indices[1][k]] = 0 
num_elem_in_row = np.array((gtest_Xu!=0).sum(axis=1)).ravel()
gtruth_Xu = gtruth_Xu[num_elem_in_row>1 , :]
gtest_Xu = gtest_Xu[num_elem_in_row>1 , :]
print (num_elem_in_row>1).sum()
cPickle.dump(gtruth_Xu, open('./data/Xu_true.pkl', 'w'))
cPickle.dump(gtest_Xu, open('./data/Xu_test.pkl', 'w'))