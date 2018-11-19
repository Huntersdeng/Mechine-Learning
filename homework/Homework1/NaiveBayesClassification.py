import numpy as np
import xlrd

def Bayes(X, Y):
    #for discrete features
    #ConditionalProb: P(X|C) 
    #dimension (good or bad, the type of feature, feature)
    #Prob :P(X)
    #dimension (the type of feature, feature)
    #for consistent features
    #NormalityCondi: the average and variance for each features in good or bad conditions
    #dimension (good or bad, the type of feature, parameters)
    #Normality : the average and variance for each features
    #dimension (the type of feature, parameters)
    ConditionalProb = np.zeros(shape=(2,6,3))
    Prob = np.zeros(shape=(6,3))
    Normality = np.zeros(shape=(2,2))
    NormalityCondi = np.zeros(shape=(2,2,2))
    n = X.shape[1]
    cnt_c = 0
    for i in range(n):
        if Y[0][i]==1:
            cnt_c += 1
    P_c = cnt_c / n
    for i in range(6):
        for j in range(n):
            if Y[0][j]==0:
                if X[i][j]==1:
                    ConditionalProb[0][i][0]+=1
                elif X[i][j]==2:
                    ConditionalProb[0][i][1]+=1
                else:
                    ConditionalProb[0][i][2]+=1
            else:
                if X[i][j]==1:
                    ConditionalProb[1][i][0]+=1
                elif X[i][j]==2:
                    ConditionalProb[1][i][1]+=1
                else:
                    ConditionalProb[1][i][2]+=1
    ConditionalProb[0] = ConditionalProb[0] / (n-cnt_c)
    ConditionalProb[1] = ConditionalProb[1] / cnt_c
    for i in range(6):
        for j in range(n):
            if X[i][j]==1:
                Prob[i][0]+=1
            elif X[i][j]==2:
                Prob[i][1]+=1
            else:
                Prob[i][2]+=1
    Prob = Prob / n
    
    data1 = X[6]
    data2 = X[7]
    Normality[0][0] = np.sum(data1) / n
    Normality[1][0] = np.sum(data2) / n
    Normality[0][1] = np.dot(data1, data1.T) / n
    Normality[1][1] = np.dot(data2, data2.T) / n

    data3 = np.zeros((1,cnt_c))
    data4 = np.zeros((1,cnt_c))
    data5 = np.zeros((1,n-cnt_c))
    data6 = np.zeros((1,n-cnt_c))
    
    cnt0 = 0
    cnt1 = 0
    for i in range(n):
        if Y[0][i]==1:
            data3[0][cnt0] = X[6][i]
            data4[0][cnt0] = X[7][i]
            cnt0+=1
        else:
            data5[0][cnt1] = X[6][i]
            data6[0][cnt1] = X[7][i]
            cnt1+=1
    NormalityCondi[1][0][0] = np.sum(data3) / cnt0
    NormalityCondi[1][1][0] = np.sum(data4) / cnt0
    NormalityCondi[0][0][0] = np.sum(data5) / cnt1
    NormalityCondi[0][1][0] = np.sum(data6) / cnt1
    NormalityCondi[1][0][1] = np.dot(data3, data3.T) / cnt0
    NormalityCondi[1][1][1] = np.dot(data4, data4.T) / cnt0
    NormalityCondi[0][0][1] = np.dot(data5, data5.T) / cnt1
    NormalityCondi[0][1][1] = np.dot(data6, data6.T) / cnt1

    parameters = {'ConditionalProb':ConditionalProb,
                  'Prob':Prob,
                  'NormalityCondi':NormalityCondi,
                  'Normality':Normality,
                  'P_c':P_c}
    return parameters

def NormalFunc(x, u, variance):
    root = -(x-u)**2/(2*variance)
    val = np.exp(root) / np.sqrt(2*3.14*variance)
    return val

def eval(X, Y,parameters):
    ConditionalProb = parameters['ConditionalProb']
    Prob = parameters['Prob']
    Normality = parameters['Normality']
    NormalityCondi = parameters['NormalityCondi']
    P_c =parameters['P_c']
    discrete = X[0:6].T
    consistent = X[6:8].T
    n = X.shape[1]
    Prob_good = np.ones((1,n))
    Prob_bad = np.ones((1,n))
    for i in range(discrete.shape[0]):
        for j in range(discrete.shape[1]):
            Prob_good[0][i] *= ConditionalProb[1][j][int(discrete[i][j])-1]
            Prob_bad[0][i] *= ConditionalProb[0][j][int(discrete[i][j])-1]
    for i in range(consistent.shape[0]):
        for j in range(consistent.shape[1]):
            Prob_good[0][i] *= NormalFunc(consistent[i][j], NormalityCondi[1][j][0], NormalityCondi[1][j][1])
            Prob_bad[0][i] *= NormalFunc(consistent[i][j], NormalityCondi[0][j][0], NormalityCondi[0][j][1])
    Prob_good *= P_c
    Prob_bad *= 1 - P_c
    eval = Prob_good>Prob_bad
    accuracy = 0
    for i in range(n):
        if eval[0][i]==Y[0][i]:
            accuracy +=1
    accuracy = accuracy / n
    print('The accuracy for test set is ',accuracy)
    
def main():
    #load the dataset
    DATA_FILE="dataset.xlsx"
    book = xlrd.open_workbook(DATA_FILE, encoding_override='utf-8')
    sheet = book.sheet_by_index(0)
    X = np.zeros(shape=(8,17))
    Y = np.zeros(shape=(1,17))
    for i in range(22,39):
        for j in range(15,23):
            X[j - 15][i - 22] = sheet.cell_value(i,j)
    for i in range(22,39):
        Y[0][i - 22] = sheet.cell_value(i,23)
    for i in range(5):
        X_test = X[:,4*i:min(4*(i+1),17)]
        Y_test = Y[:,4*i:min(4*(i+1),17)]
        X_train = np.delete(X, np.s_[4*i:min(4*(i+1),17)], axis=1)
        Y_train = np.delete(Y, np.s_[4*i:min(4*(i+1),17)], axis=1)
        #print(X_train,'\n',Y_train,'\n',X_test,'\n',Y_test)
        parameters = Bayes(X_train,Y_train)
        eval(X_test, Y_test, parameters)
    #print('1',ConditionalProb,'\n', '2',Prob, '\n','3', Normality,'\n','4', NormalityCondi)

if __name__ == "__main__":
    main()