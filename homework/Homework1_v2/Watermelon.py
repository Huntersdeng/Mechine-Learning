import numpy as np
from sklearn import svm

def NormalFunc(x, u, variance):
    root = -(x-u)**2/(2*variance)
    val = np.exp(root) / np.sqrt(2*3.14*variance)
    return val

class Linearclf:
    def __init__(self, input_dim):
        self.dim = input_dim
    
    def train(self, X, Y):
        n = X.shape[0]
        e = np.ones(shape=(n,1))
        X = np.concatenate((X,e), axis=1)
        W = np.zeros(shape=(self.dim+1,1))
        W = np.dot(X.T,X)
        W = np.linalg.inv(W)
        W = np.dot(W,X.T)
        W = np.dot(W,Y)
        #print('Training complete')
        return W

    def test(self, X,Y,W):
        n = X.shape[0]
        e = np.ones(shape=(n,1))
        X = np.concatenate((X,e), axis=1)
        y_predict = np.dot(X,W)
        y_predict = y_predict>0.5
        #print('Predict result is',y_predict)
        accuracy = np.count_nonzero(y_predict==Y)
        #print('Test accuracy is',accuracy/X.shape[0])
        return accuracy

class Naivebayesclf:
    def __init__(self, input_dim):
        self.dim = input_dim

    def train(self, data):
        n = data.shape[0]
        c1 = []
        c0 = []
        cnt = 0
        for i in range(n):
            if data[i,8]==1:
                c1.append(data[i,0:8])
                cnt+=1
            else:
                c0.append(data[i,0:8])
        P_c = cnt / n
        data0 = np.array(c0)
        data1 = np.array(c1)
        data0_dense = data0[:,6:8]
        data1_dense = data1[:,6:8]
        sparse = np.zeros((2,6,3))
        dense = np.zeros((2,2,2))
        for i in range(6):
            sparse[0,i,0]=np.count_nonzero(data0[:,i]==1)
            sparse[0,i,1]=np.count_nonzero(data0[:,i]==2)
            sparse[0,i,2]=np.count_nonzero(data0[:,i]==3)
            sparse[1,i,0]=np.count_nonzero(data1[:,i]==1)
            sparse[1,i,1]=np.count_nonzero(data1[:,i]==2)
            sparse[1,i,2]=np.count_nonzero(data1[:,i]==3)
        sparse[0] = sparse[0]/(n-cnt)
        sparse[1] = sparse[1]/cnt
        dense[:,0,0] = np.mean(data0_dense,axis=0)
        dense[:,0,1] = np.var(data0_dense,axis=0)
        dense[:,1,0] = np.mean(data1_dense,axis=0)
        dense[:,1,1] = np.var(data1_dense,axis=0)
        parameters={'sparse':sparse,
                    'dense':dense,
                    'P_c':P_c}
        #print('Training complete')
        return parameters

    def test(self, data, parameters):
        sparse = parameters['sparse']
        dense = parameters['dense']
        P_c = parameters['P_c']
        n = data.shape[0]
        data_dense = data[:,6:8]
        data_sparse = data[:,0:6]
        Prob_good = np.ones((n,))
        Prob_bad = np.ones((n,))
        for i in range(data_sparse.shape[0]):
            for j in range(data_sparse.shape[1]):
                Prob_good[i] *= sparse[1][j][int(data_sparse[i][j])-1]
                Prob_bad[i] *= sparse[0][j][int(data_sparse[i][j])-1]
        for i in range(data_dense.shape[0]):
            for j in range(data_dense.shape[1]):
                Prob_good[i] *= NormalFunc(data_dense[i][j], dense[1][j][0], dense[1][j][1])
                Prob_bad[i] *= NormalFunc(data_dense[i][j], dense[0][j][0], dense[0][j][1])
        Prob_good *= P_c
        Prob_bad *= 1 - P_c
        eval = Prob_good>Prob_bad
        #print(eval)
        cnt = 0
        for i in range(n):
            if eval[i]==data[i,8]:
                cnt +=1
        accuracy = cnt / n
        #print('The accuracy for test set  is ',accuracy)
        return cnt

class SVM:
    def __init__(self, input_dim, kernel):
        self.dim = input_dim
        self.clf = svm.SVC(kernel=kernel)

    def train(self, X_train, Y_train):
        self.clf.fit(X_train,Y_train)
    
    def test(self, X_test, Y_test):
        predict = self.clf.predict(X_test)
        accuracy = np.count_nonzero(predict==Y_test)
        return accuracy


class lnregression:
    def __init__(self, input_dim):
        self.dim = input_dim
    
    def train(self, X, Y, learningrate, iter):
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
        #print('Training complete')

        return W,b

    def test(self, X, Y, W, b):
        n = X.shape[1]
        _y = np.zeros(shape=(1,n))
        for i in range(Y.shape[0]):
            _y[0,i]=Y[i]
        y = np.dot(W.T,X) + b
        y_predict = 1 / (1 + np.exp(-y))
        y_predict = y_predict>0.5
        #print(Y,y_predict)
        accuracy = np.count_nonzero(y_predict==_y)
        #print('The accuracy for test set is ',accuracy)
        return accuracy

    
if __name__ == "__main__":
    data = np.loadtxt(open("dataset.csv","rb"),delimiter=",",skiprows=0)
    dim = data.shape[1] - 1
    cnt_lclf = 0
    cnt_nbc = 0
    cnt_svm = [0,0,0,0]
    cnt_lnclf = 0
    lclf = Linearclf(dim)
    nbclf = Naivebayesclf(dim)
    k = ['linear','poly','rbf','sigmoid']
    lnclf = lnregression(dim)
    print('In k=5 cross validation')
    for i in range(5):
        test = data[int(17*i/5):int(17*(i+1)/5)]
        train = np.delete(data, np.s_[int(17*i/5),int(17*(i+1)/5)], axis=0)
        X_test = test[:,0:8]
        Y_test = test[:,8]
        X_train = train[:,0:8]
        Y_train = train[:,8]
        W = lclf.train(X_train, Y_train)
        cnt_lclf += lclf.test(X_test, Y_test, W)
        parameters = nbclf.train(train)
        cnt_nbc+=nbclf.test(test, parameters)
        j = 0
        for kernel in k:
            Svm = SVM(dim,kernel)
            Svm.train(X_train, Y_train)
            cnt_svm[j]+=Svm.test(X_test, Y_test)
            j+=1
        W, b = lnclf.train(X_train.T, Y_train, 0.01, 1000)
        cnt_lnclf+=lnclf.test(X_test.T, Y_test, W, b)
    print('Total accuracy for linear classifier:',cnt_lclf/17)
    print('Total accuracy for naive bayes classifier:',cnt_nbc/17)
    print('Total accuracy for SVM of kernel \'linear\':',cnt_svm[0]/17)
    print('Total accuracy for SVM of kernel \'poly\':',cnt_svm[1]/17)
    print('Total accuracy for SVM of kernel \'rbf\':',cnt_svm[2]/17)
    print('Total accuracy for SVM of kernel \'sigmoid\':',cnt_svm[3]/17)
    print('Total accuracy for logistic regression:',cnt_lnclf/17)