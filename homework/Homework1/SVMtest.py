from sys import path
path.append(r'd:\\MyDocument\\Study_makes_me_happy\\ManchineLearning\\libsvm\\python')
from svmutil import *


# Construct problem in python format
# Dense data
y, x = [1,-1], [[1,0,1], [-1,0,-1]]
# Sparse data
y, x = [1,-1], [{1:1, 3:1}, {1:-1,3:-1}]
prob  = svm_problem(y, x)
param = svm_parameter('-t 0 -c 4 -b 1')
m = svm_train(prob, param)

# Precomputed kernel data (-t 4)
# Dense data
y, x = [1,-1], [[1, 2, -2], [2, -2, 2]]
# Sparse data
y, x = [1,-1], [{0:1, 1:2, 2:-2}, {0:2, 1:-2, 2:2}]
# isKernel=True must be set for precomputed kernel
prob  = svm_problem(y, x, isKernel=True)
param = svm_parameter('-t 4 -c 4 -b 1')
m = svm_train(prob, param)
# For the format of precomputed kernel, please read LIBSVM README.


# Other utility functions
#svm_save_model('heart_scale.model', m)
#m = svm_load_model('heart_scale.model')
#p_label, p_acc, p_val = svm_predict(y, x, m, '-b 1')
#ACC, MSE, SCC = evaluations(y, p_label)