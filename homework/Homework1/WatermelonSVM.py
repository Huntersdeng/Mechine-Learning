import numpy as np 
import xlrd
from sys import path
path.append(r'd:\\MyDocument\\Study_makes_me_happy\\ManchineLearning\\libsvm\\python')
from svmutil import *

DATA_FILE="dataset.xlsx"
book = xlrd.open_workbook(DATA_FILE, encoding_override='utf-8')
sheet = book.sheet_by_index(0)
X = np.zeros(shape=(8,17))
Y = np.zeros(shape=(17,))
for i in range(22,39):
    for j in range(15,23):
        X[j - 15][i - 22] = sheet.cell_value(i,j)
for i in range(22,39):
    if sheet.cell_value(i,23)==1:
        Y[i - 22] = 1
    else:
        Y[i - 22] = -1
X_train = X[:,3:17]
Y_train = Y[3:17]
X_test = X[:,0:3]
Y_test = Y[0:3]
prob  = svm_problem(Y_train, X_train.T)
param = svm_parameter('-s 1 -t 0 -b 1')
m = svm_train(prob, param)
p_label, p_acc, p_val = svm_predict(Y_test, X_test.T, m, '-b 1')