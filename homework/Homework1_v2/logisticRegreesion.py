import numpy as np 

def train(X, Y, learningrate, iter):
    n = X.shape[1]
    m = X.shape[0]
    _y = np.zeros(shape=(1,n))
    for i in range(Y.shape[0]):
        _y[0,i]=Y[i]
    W = np.random.randn(m,1)*0.01
    b = np.zeros(1)
    for i in range(iter):
        y = np.dot(W.T,X) + b
        p = 1 / (1 + np.exp(-y))
        dW = -np.dot(X,_y.T) + np.dot(X,p.T)
        db = np.sum(p-_y)
        W = W - dW * learningrate
        b = b - db * learningrate
    print('Training complete')

    return W,b

def test(X, Y, W, b):
    n = X.shape[1]
    _y = np.zeros(shape=(1,n))
    for i in range(Y.shape[0]):
        _y[0,i]=Y[i]
    y = np.dot(W.T,X) + b
    y_predict = 1 / (1 + np.exp(-y))
    y_predict = y_predict>0.5
    print(Y,y_predict)
    accuracy = np.count_nonzero(y_predict==_y)
    accuracy = accuracy / Y.shape[0]
    print('The accuracy for test set is ',accuracy)
    return accuracy

if __name__=='__main__':
    data = np.loadtxt(open("dataset.csv","rb"),delimiter=",",skiprows=0)
    X = data[:,0:8].T
    Y = data[:,8]
    X_train = X[:,0:14]
    X_test = X[:,14:17]
    Y_train = Y[0:14]
    Y_test = Y[14:17]
    W, b = train(X_train, Y_train, 0.1, 100)
    accuracy = test(X_test, Y_test, W, b)
