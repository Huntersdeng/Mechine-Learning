import scipy.io as sio
import numpy as np
from random import shuffle   
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.keras as K


class Mnist:
    def __init__(self, input_shape, optimizer, loss, metrics):
        X_input = K.Input(input_shape)

        X = K.layers.Conv2D(filters=32, kernel_size=(2,2), activation='relu', kernel_initializer=tf.contrib.layers.xavier_initializer())(X_input)
        X = K.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(X)
        X = K.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(2,2),activation='relu', kernel_initializer=tf.contrib.layers.xavier_initializer())(X)
        X = K.layers.MaxPool2D(pool_size=(2,2))(X)
        X = K.layers.Flatten()(X)
        X = K.layers.Dense(1024, activation='relu')(X)
        X = K.layers.Dropout(0.6)(X)
        X = K.layers.Dense(10, activation='softmax')(X)
        self.model = K.Model(inputs = X_input, outputs = X, name='mnist')
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        self.model.summary()

    def train(self, X_train, y_train, batch_size, epochs, save_file=None):
        train_costs = []
        train_accuracy = []
        test_costs = []
        test_accuracy = []
        for i in range(epochs):
            self.model.fit(X_train, y_train, batch_size=batch_size)
            train_loss, train_acc = self.model.evaluate(X_train, y_train, batch_size=batch_size)
            train_costs.append(train_loss)
            train_accuracy.append(train_acc)
            test_loss, test_acc = self.model.evaluate(X_test, y_test, batch_size=batch_size)
            print('Test accuracy:',test_acc)
            test_costs.append(test_loss)
            test_accuracy.append(test_acc)
        if save_file!=None:
            self.model.save_weights(save_file)
        print('Train accuracy:', train_acc)
        print('Test accuracy:',test_acc)

        plt.plot(np.squeeze(train_costs),color='red',label='training cost')
        plt.plot(np.squeeze(test_costs),color='blue',label='testing cost')
        plt.ylabel('cost')
        plt.xlabel('iterations')
        plt.title('Change of cost')
        plt.legend(loc='upper right')
        # plt.savefig('cost_mnist.png')
        plt.show()

        plt.plot(np.squeeze(train_accuracy),color='red',label='training accuracy')
        plt.plot(np.squeeze(test_accuracy),color='blue',label='testing accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('iterations')
        plt.title('Change of accuracy')

        plt.legend(loc='lower right')
        # plt.savefig('accuracy_mnist.png')
        plt.show()

    def eval(self, X_test, y_test, batch_size, model_weight):
        self.model.load_weights(model_weight)
        test_loss, test_acc = self.model.evaluate(X_test, y_test, batch_size=batch_size)
        print('Test accuracy:',test_acc)


if __name__ == '__main__':
    load_fn = 'MNIST.mat'
    load_data = sio.loadmat(load_fn)
    X = load_data['fea']
    y = load_data['gnd']
    x = X[0].reshape(28,28) # 变换成28x28的矩阵
    plt.imshow(x) #显示图像
    print('shape of data is',X.shape)
    print('shape of labels is',y.shape)
    data = np.concatenate((X, y),axis=1)
    shuffle(data)
    X = data[:,0:784]
    X = X / 255
    y = (np.arange(10)==data[:,-1][:,None]).astype(np.integer) 
    X_train = X[0:60000].reshape(60000,28,28,1)
    y_train = y[0:60000]
    X_test = X[60000:].reshape(10000,28,28,1)
    y_test = y[60000:]
    print('shape of train data is',X_train.shape)
    print('shape of train labels is',y_train.shape)
    print('shape of test data is',X_test.shape)
    print('shape of test labels is',y_test.shape)
    mnist = Mnist(X_train.shape[1:], optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # mnist.train(X_train, y_train, 64, 1)
    mnist.eval(X_test, y_test, 64, 'model_weight_normal.h5')
    