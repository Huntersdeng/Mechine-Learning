import scipy.io as sio
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time

class iris():
    def __init__(self):
        pass
    
    
    def train(self, X_train, y_train, X_test, y_test, n_h, activate, learningrate=0.01, iteration = 100, save_print=True):
        #input : 
        # training data : X_train, y_train
        #  testing data  : X_test, y_test
        #  number of hidden unit : n_h
        #  activate function : activate
        #  learningrate, default 0.01
        #  iteration, default 100
        #  whether to save result to file: save_result, default True
        np.random.seed(0)
        out = open('result', 'a')
        m_x = X_train.shape[1]
        n = X_train.shape[0]
        m_y = y_train.shape[1]
        n_output = 3
        train_costs = [] # a list to store the cost in every iterations
        test_costs = []
        train_accuracy = []
        test_accuracy = []
        # define the placeholder
        X = tf.placeholder(tf.float64, shape=(None, m_x), name='input_data')
        y = tf.placeholder(tf.int32, shape=(None, m_y), name='input_label')
        
        # initialize the parameters
        W1 = tf.Variable(np.random.randn(m_x, n_h)*0.01, name='W1', dtype=tf.float64)
        b1 = tf.Variable(0, dtype=tf.float64, name='b1')
        W2 = tf.Variable(np.random.randn(n_h, n_output)*0.01, name='W2', dtype=tf.float64)
        b2 = tf.Variable(0, dtype=tf.float64, name='b2')

        # define the graph
        Z1 = tf.add(tf.matmul(X, W1), b1, name='Z1')
        if activate=='relu':
            A1 = tf.nn.relu(Z1, name='A1')
        if activate=='sigmoid':
            A1 = tf.nn.sigmoid(Z1, name='A1')
        if activate=='tanh':
            A1 = tf.nn.tanh(Z1, name='A1')
        Z2 = tf.add(tf.matmul(A1, W2), b2, name='Z2')
        y_predict = tf.nn.softmax(Z2)
        optimizer = tf.train.GradientDescentOptimizer(learningrate)
        loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(y, Z2))
        train = optimizer.minimize(loss)

        init = tf.global_variables_initializer()
        
        correct_prediction = tf.equal(tf.argmax(y_predict,axis=1), tf.argmax(y,axis=1))
            
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(init)
            start = time.time()
            saver.restore(sess, "/tmp/model.ckpt")
            for i in range(iteration):
                train_cost, _ = sess.run([loss, train], feed_dict={X:X_train, y:y_train})
                test_cost, _ = sess.run([loss, train], feed_dict={X:X_test, y:y_test})
                # Print the cost every iteration
                if i % 100 == 0:
                    print ("Cost after iteration %i: %f" % (i, train_cost))
                if i%5 == 0:
                    train_costs.append(train_cost)
                    test_costs.append(test_cost)
                    train_accuracy.append(accuracy.eval({X: X_train, y: y_train}))
                    test_accuracy.append(accuracy.eval({X: X_test, y: y_test}))
            save_path = saver.save(sess, "/tmp/model.ckpt")
            done = time.time()
            print("It costs "+str((done-start)/10)+' ms per step using '+str(activate))
            plt.plot(np.squeeze(train_costs),color='red',label='training cost')
            plt.plot(np.squeeze(test_costs),color='blue',label='testing cost')
            plt.ylabel('cost')
            plt.xlabel('iterations')
            plt.title("Learning rate=" + str(learningrate) + ' hidden units=' + str(n_h) + ' activate function=' + activate)
            plt.legend(loc='upper right')
            # plt.savefig('cost_' + activate + '_' + str(n_h) + '.png')
            plt.show()
            
            plt.plot(np.squeeze(train_accuracy),color='red',label='training accuracy')
            plt.plot(np.squeeze(test_accuracy),color='blue',label='testing accuracy')
            plt.ylabel('accuracy')
            plt.xlabel('iterations')
            plt.title("Learning rate=" + str(learningrate) + ' hidden units=' + str(n_h) + ' activate function=' + activate)
            plt.legend(loc='lower right')
            # plt.savefig('accuracy_' + activate + '_' + str(n_h) + '.png')
            plt.show()
            
#             print(Z2.eval({X: X_train, y: y_train}))
#             print(tf.argmax(Z2,axis=1).eval({X: X_train, y: y_train}))
            print('The accuracy with activate function %s, hidden units %i, learning rate %.3f'%(activate, n_h,learningrate))
            print("Train Accuracy:", accuracy.eval({X: X_train, y: y_train}))
            print("Test Accuracy:", accuracy.eval({X: X_test, y: y_test}))
            # print('The accuracy with activate function %s, hidden units %i, learning rate %.3f'%(activate, n_h,learningrate), file=out)
            # print("Train Accuracy:", accuracy.eval({X: X_train, y: y_train}), file=out)
            # print("Test Accuracy:", accuracy.eval({X: X_test, y: y_test}), file=out)


if __name__ == '__main__':
    load_fn = 'iris.mat'
    load_data = sio.loadmat(load_fn)
    # load_data is a dict with key named samples, labels, __globals__, __version__, __header__
    X = load_data['samples']
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

    model = iris()
    
    model.train(X_train, y_train, X_test, y_test, 5, 'relu', 0.01, 1000)
    done = time.time()