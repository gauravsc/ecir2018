import scipy.io
import numpy as np
import cPickle

input_filename = "./data/Xu.mat"
m1 = scipy.io.loadmat(input_filename)['Xu']
m1 = m1.transpose()
m1[m1 >0] = 1
cPickle.dump(m1, open('./data/Xu.pkl', 'w'))
input_filename = "./data/Xs.mat"
m2 = scipy.io.loadmat(input_filename)['Xs']
cPickle.dump(m2, open('./data/Xs.pkl', 'w'))	