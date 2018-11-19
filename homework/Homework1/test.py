import tensorflow as tf
import numpy as np
import xlrd

def model(X_train, Y_train, X_test, Y_test, learningrate, iterations):
    
    X = tf.placeholder(tf.float64,shape=(8,None), name='input')
    Y = tf.placeholder(tf.float64,shape=(1,None), name='output')
    W1 = tf.Variable(np.random.randn(3, X.shape[0]) * 0.01, name='weight1')
    b1 = tf.Variable(np.random.randn(1) * 0.01, name='bias1')
    W2 = tf.Variable(np.random.randn(1, 3) * 0.01, name='weight2')
    b2 = tf.Variable(np.random.randn(1) * 0.01, name='bias2')
    parameters={'W1':W1,
                'W2':W2,
                'b1':b1,
                'b2':b2}

    Z1 = tf.matmul(W1, X) + b1
    A1 = tf.nn.relu(Z1)
    Z2 = tf.matmul(W2, A1) + b2
    loss = tf.losses.sigmoid_cross_entropy(Y, Z2)
    optimizer = tf.train.AdamOptimizer(learningrate).minimize(loss)
    writer = tf.summary.FileWriter('./graphs', tf.get_default_graph())
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(iterations):
            _, cost =sess.run([optimizer, loss], feed_dict={X: X_train, Y: Y_train})
            if i % 10 == 0:
                print ("Cost after iteration %i: %f" , i, cost)
        parameters = sess.run(parameters)
        # Calculate the correct predictions
        correct_prediction = tf.equal(tf.argmax(Z2), tf.argmax(Y))

        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        print("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
        print("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))
    writer.close()
    return parameters


def main():
    #load the dataset
    DATA_FILE="dataset.xlsx"
    book = xlrd.open_workbook(DATA_FILE, encoding_override='utf-8')
    sheet = book.sheet_by_index(0)
    X_train = np.zeros(shape=(8,14))
    Y_train = np.zeros(shape=(1,14))
    X_test = np.zeros(shape=(8,3))
    Y_test = np.zeros(shape=(1,3))
    for i in range(22,36):
        for j in range(15,23):
            X_train[j - 15][i - 22] = sheet.cell_value(i,j)
    for i in range(22,36):
        Y_train[0][i - 22] = sheet.cell_value(i,23)

    for i in range(36,39):
        for j in range(15,23):
            X_test[j - 15][i - 36] = sheet.cell_value(i,j)
    for i in range(36,39):
        Y_test[0][i - 36] = sheet.cell_value(i,23)
    #train the parameters
    parameters = model(X_train, Y_train, X_test, Y_test, 0.08, 1000)

if __name__ == "__main__":
    main()