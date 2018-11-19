from sklearn import svm
import numpy as np

data = np.loadtxt(open("dataset.csv","rb"),delimiter=",",skiprows=0)
X = data[:,0:8]
Y = data[:,8]
X_train = X[0:14]
X_test = X[14:17]
Y_train = Y[0:14]
Y_test = Y[14:17]
k = ['linear','poly','rbf','sigmoid']
for kernel in k:
    clf = svm.SVC(kernel=kernel)
    clf.fit(X_train,Y_train)
    predict = clf.predict(X_test)
    print(predict)
    accuracy = np.count_nonzero(predict==Y_test)
    print('For kernel', kernel, ',','the accuracy for test set is ',accuracy/X_test.shape[0])