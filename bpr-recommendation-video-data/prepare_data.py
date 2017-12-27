import numpy as np
import cPickle

input_filename = "./data/Xu.pkl"
Xu = cPickle.load(open(input_filename, 'r'))
Xu = Xu.tocsr()
Xu[Xu>0] = 1
cPickle.dump(Xu, open('./data/Xu.pkl', 'w'))	