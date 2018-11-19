import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import time

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(Z):
    m = Z.shape[0]
    return np.exp(Z)/np.sum(np.exp(Z),axis=1).reshape(m,1)

class iris:
    # set the nodes in the hidden layer
    def __init__(self, n_h):
        self._h = n_h

    # training 
    # return the trained parameters
    def fit(self, X_train, y_train, learningrate, epochs):
        input_shape = X_train.shape[1]
        output_shape = y_train.shape[1]
        m = X_train.shape[0]

        # initialize all variables
        W1 = np.random.randn(input_shape, self._h) 
        b1 = np.zeros((1, self._h))
        W2 = np.random.randn(self._h, output_shape) 
        b2 = np.zeros((1, output_shape))
        parameters = {'W2':W2, 'b2':b2, 'W1':W1, 'b1':b1}
        train_costs = []
        test_costs = []
        train_accuracy = []
        test_accuracy = []
        start = time.time()
        for i in range(epochs):
            # forward propagation
            Z1 = np.dot(X_train, W1) + b1
            A1 = sigmoid(Z1)
            Z2 = np.dot(A1, W2) + b2
            A2 = softmax(Z2)
            
            # loss function using crosstropy
            loss = -np.sum(y_train*np.log(A2))/m
            loss = np.squeeze(loss)
            train_costs.append(loss)

            # back propagation
            # To simplify the names of variables, use the denominator to represent the derivative 
            # For example, dx means dl/dx
            dZ2 = A2 - y_train
            dW2 = np.dot(A1.T, dZ2) / m
            db2 = np.sum(dZ2, axis=0) / m
            dZ1 = np.dot(dZ2, W2.T)*A1*(1-A1)
            dW1 = np.dot(X_train.T, dZ1) / m
            db1 = np.sum(dZ1, axis=0) / m

            # update parameters
            W2 = W2 - dW2 * learningrate
            b2 = b2 - db2 * learningrate
            W1 = W1 - dW1 * learningrate
            b1 = b1 - db1 * learningrate
            parameters['W2'] = W2
            parameters['b2'] = b2
            parameters['W1'] = W1
            parameters['b1'] = b1
            
            # compute the accuracy
            train_acc = np.mean(np.equal(np.argmax(A2, axis=1), np.argmax(y_train, axis=1)))
            train_accuracy.append(train_acc)

            # evaluate the accurcy and loss in test set
            test_acc, test_cost = self.eval(X_test, y_test, parameters)
            test_accuracy.append(test_acc)
            test_costs.append(test_cost)

            # print the loss
            if i%1000==0:
                print ("Cost after iteration %i: %f" % (i, loss))
        done = time.time()
        print("It costs "+str(1000*(done-start)/epochs)+' ms per step')
        # compute the final accuracy
        train_acc, train_cost = self.eval(X_train, y_train, parameters)
        test_acc, test_cost = self.eval(X_test, y_test, parameters)
        train_accuracy.append(train_acc)
        test_accuracy.append(test_acc)
        train_costs.append(train_cost)
        test_costs.append(test_cost)

        #plot the change of cost and accuracy
        plt.plot(np.squeeze(train_costs),color='red',label='training cost')
        plt.plot(np.squeeze(test_costs),color='blue',label='testing cost')
        plt.ylabel('cost')
        plt.xlabel('iterations')
        plt.title("Learning rate=" + str(learningrate) + ' hidden units=' + str(self._h))
        plt.legend(loc='upper right')
        plt.savefig('cost_' + str(self._h) + '_diy.png')
        plt.show()
        
        plt.plot(np.squeeze(train_accuracy),color='red',label='training accuracy')
        plt.plot(np.squeeze(test_accuracy),color='blue',label='testing accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('iterations')
        plt.title("Learning rate=" + str(learningrate) + ' hidden units=' + str(self._h))
        plt.legend(loc='lower right')
        plt.savefig('accuracy_' + str(self._h) + '_diy.png')
        plt.show()
        print('Train accuracy:', train_acc)
        print("Test Accuracy:", test_acc)
        return parameters

    def eval(self, X, y, parameters):
        m = y.shape[0]
        # unpack parameters
        W2 = parameters['W2']
        b2 = parameters['b2']
        W1 = parameters['W1']
        b1 = parameters['b1']

        # forward propagation
        Z1 = np.dot(X, W1) + b1
        A1 = sigmoid(Z1)
        Z2 = np.dot(A1, W2) + b2
        A2 = softmax(Z2)

        # loss function using crosstropy
        loss = -np.sum(y*np.log(A2))/m
        loss = np.squeeze(loss)

        accuracy = np.mean(np.equal(np.argmax(A2, axis=1), np.argmax(y, axis=1)))
        
        return accuracy, loss

if __name__ == '__main__':
    np.random.seed(1)
    load_fn = 'iris.mat'
    load_data = sio.loadmat(load_fn)
    # load_data is a dict with key named samples, labels, __globals__, __version__, __header__
    X = load_data['samples']
    X = X / 8
    y = load_data['labels']
    y_shape = load_data['labels'].shape
    X_shape = load_data['samples'].shape
    print('shape of data is',X_shape)
    print('shape of labels is',y_shape)
    X_train = X[0:int(X_shape[0]*0.8)]
    y_train = y[0:int(y_shape[0]*0.8)]
    X_test = X[int(X_shape[0]*0.8):]
    y_test = y[int(y_shape[0]*0.8):]
    print('shape of train data is',X_train.shape)
    print('shape of train labels is',y_train.shape)
    print('shape of test data is',X_test.shape)
    print('shape of test labels is',y_test.shape)
    model = iris(5)
    model.fit(X_train, y_train, 0.2, 5000)
    

    
    