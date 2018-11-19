import numpy as np 
from sklearn import svm

def train(X, Y):
    n = X.shape[1]
    W = np.zeros(shape=(n,1))
    W = np.dot(X.T,X)
    W = np.linalg.inv(W)
    W = np.dot(W,X.T)
    W = np.dot(W,Y)
    print('Training complete')
    return W

def test(X,Y,W):
    y_predict = np.dot(X,W)
    y_predict = y_predict>0.5
    print('Predict result is',y_predict)
    accuracy = (y_predict==Y)
    print('Test accuracy is',np.count_nonzero(accuracy)/X.shape[0])

if __name__ == "__main__":
    X = np.loadtxt(open("dataset.csv","rb"),delimiter=",",skiprows=0)
    Y = np.zeros(shape=(17,))
    for i in range(X.shape[0]):
        Y[i] = X[i,8]
    X[:,8]=1
    # print(X,Y)
    X_train = X[0:14]
    Y_train = Y[0:14]
    X_test = X[14:17]
    Y_test = Y[14:17]
    W = train(X_train, Y_train)
    test(X_test, Y_test, W)

