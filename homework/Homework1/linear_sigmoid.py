import numpy as np
import xlrd

#sigmod function
def sigmoid(z):
    Y = 1 / (1 + np.exp(-z))
    return Y

#initialize parameters
def init(X):
    W = np.random.randn(X.shape[0], 1) * 0.01
    b = np.random.randn(1) * 0.01
    parameters = {'W':W, 'b':b}
    return parameters

#forword propagation
def forward_propagation(X, parameters):
    W = parameters['W']
    b = parameters['b']
    Z = np.dot(W.T,X)+b
    A = sigmoid(Z)
    cache = {'Z':Z, 'A':A}
    return A,cache

#cost function use binary_crossentropy loss function
def compute_cost(cache, Y_label):
    Y = cache['A']
    n = Y_label.shape[0]
    cost = -(Y_label*np.log(Y) + (1-Y_label)*np.log(1-Y))
    cost = np.sum(cost) / n
    cost = np.squeeze(cost)     # makes sure cost is the dimension we expect. 
                                # E.g., turns [[17]] into 17 
    return cost

def backward_propagation(cache, X, Y_label):
    
    Z = cache['Z']
    A = cache['A']
    n = Y_label.shape[0]

    #compute the derivative of the parameters
    #dZ for dJ/dZ, dW for dJ/dW, db for dJ/db
    dZ = A - Y_label
    dW = (1 / n) * np.dot(dZ, X.T) 
    db = (1 / n) * np.sum(dZ, axis=1, keepdims=True) 

    grads = {'dW':dW, 'db':db}
    return grads

#default learning rate is 0.01
def update_parameters(grads, parameters, learningrate):

    W = parameters['W']
    b = parameters['b']
    dW = grads['dW']
    db = grads['db']

    W = W - dW.T * learningrate
    b = b - db * learningrate
    parameters = {'W':W, 'b':b}
    return parameters

def train(X, Y, learningrate, iterations):

    parameters = init(X)
    W = parameters['W']
    b = parameters['b']
    for i in range(0, iterations):
        A, cache = forward_propagation(X, parameters)
        cost = compute_cost(cache, Y)
        grads = backward_propagation(cache, X, Y)
        parameters = update_parameters(grads, parameters, learningrate)
        
        # Print the cost every 100 iterations
        if i % 10==0:
             print ("Cost after iteration %i: %f" , i, cost)

    return parameters

def test(X_test, Y_test, parameters):
    A, cache = forward_propagation(X_test, parameters)
    A = A > 0.5
    cnt = 0
    for i in range(0,Y_test.shape[0]):
        if(A[0][i]==Y_test[i]):
            cnt = cnt + 1
    return cnt / Y_test.shape[0]

def main():
    #load the dataset
    DATA_FILE="dataset.xlsx"
    book = xlrd.open_workbook(DATA_FILE, encoding_override='utf-8')
    sheet = book.sheet_by_index(0)
    X_train = np.zeros(shape=(8,14))
    Y_train = np.zeros(shape=(14,))
    X_test = np.zeros(shape=(8,3))
    Y_test = np.zeros(shape=(3,))
    for i in range(22,36):
        for j in range(15,23):
            X_train[j - 15][i - 22] = sheet.cell_value(i,j)
    for i in range(22,36):
        Y_train[i - 22] = sheet.cell_value(i,23)

    for i in range(36,39):
        for j in range(15,23):
            X_test[j - 15][i - 36] = sheet.cell_value(i,j)
    for i in range(36,39):
        Y_test[i - 36] = sheet.cell_value(i,23)
    #train the parameters
    parameters = train(X_train, Y_train, 0.2, 10000)

    #test
    accuracy = test(X_test, Y_test, parameters)
    print(accuracy)

if __name__ == "__main__":
    main()
