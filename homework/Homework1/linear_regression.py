import numpy as np
import xlrd

#load the dataset
DATA_FILE="dataset.xlsx"
book = xlrd.open_workbook(DATA_FILE, encoding_override='utf-8')
sheet = book.sheet_by_index(0)
X = np.zeros(shape=(8,17))
Y = np.zeros(shape=(17,))
for i in range(22,39):
    for j in range(15,23):
        X[j - 15][i - 22] = sheet.cell_value(i,j)
for i in range(22,39):
    Y[i - 22] = sheet.cell_value(i,23)
X_train = X[:,0:14]
Y_train = Y[0:14]
X_test = X[:,14:17]
Y_test = Y[14:17]

#initialize W
n = X.shape[0]
m = X.shape[1]
W = np.random.randn(1, n)*0.01

for iter in range(2000):
    #forward propagation
    Z = np.dot(W,X_train)

    #loss function
    loss = np.dot((Z-Y_train),(Z-Y_train).T)
    loss = np.squeeze(loss)

    #backward propagation
    dW = 2 * np.dot((Z-Y_train),X_train.T) / n
    #update W
    learningrate = 0.02
    W = W - dW * learningrate

    # Print the cost every 100 iterations
    if iter % 10==0:
        print ("Cost after iteration %i: %f" , iter, loss)

#test
Z = np.dot(W,X_test)
Z = Z > 0.5
accuracy = 0
for i in range(Y_test.shape[0]):
    if(Z[0][i]==Y_test[i]):
        accuracy = accuracy + 1
accuracy /= Y_test.shape[0]
print(accuracy)


