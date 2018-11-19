import numpy as np

def NormalFunc(x, u, variance):
    root = -(x-u)**2/(2*variance)
    val = np.exp(root) / np.sqrt(2*3.14*variance)
    return val

def Train(data):
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
    print('Training complete')
    return parameters

def Test(data, parameters):
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
    print(eval)
    cnt = 0
    for i in range(n):
        if eval[i]==data[i,8]:
            cnt +=1
    accuracy = cnt / n
    print('The accuracy for test set  is ',accuracy)
    return cnt

if __name__ == "__main__":
    data = np.loadtxt(open("dataset.csv","rb"),delimiter=",",skiprows=0)
    cnt = 0
    for i in range(5):
        test = data[int(17*i/5):int(17*(i+1)/5)]
        train = np.delete(data, np.s_[int(17*i/5),int(17*(i+1)/5)], axis=0)
        parameters = Train(train)
        cnt+=Test(test, parameters)
    print('Total accuracy:',cnt/17)

    