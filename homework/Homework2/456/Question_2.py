# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 18:58:31 2018

@author: Qiming Liu, seiee of SJTU
"""

import tensorflow as tf
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

def data_import(path):
    data_dict = sio.loadmat(path)
    samples = data_dict['fea']
    labels = data_dict['gnd']
    
    permutation = np.random.permutation(70000)
    samples = samples[permutation]
    labels = labels[permutation]
    
    training_samples = samples[:60000].reshape(-1, 28, 28, 1)
    training_labels = labels[:60000].reshape(-1)
    testing_samples = samples[60000:].reshape(-1, 28, 28, 1)
    testing_labels = labels[60000:].reshape(-1)
    sample_num, row, col, channel = training_samples.shape
    return (training_samples, training_labels, testing_samples, testing_labels, sample_num, row, col, channel)

def Network(row, col, channel, learning_rate):
    x = tf.placeholder(tf.float32, shape=(None, row, col, channel), name = 'x')
    y = tf.placeholder(tf.int64, shape=(None), name = 'y')
    select_prob = tf.placeholder(tf.float32, name = 'select_prob')
    
    conv1 = tf.layers.conv2d(x, 16, 3, padding = 'same', activation = tf.nn.relu, name = 'conv1')
    conv2 = tf.layers.conv2d(conv1, 16, 3, padding = 'same', activation = tf.nn.relu, name = 'conv2')
    pool1 = tf.layers.max_pooling2d(conv2, 2, 2, name = 'pool1')
    
    conv3 = tf.layers.conv2d(pool1, 32, 4, padding = 'same', activation = tf.nn.relu, name = 'conv3')
    conv4 = tf.layers.conv2d(conv3, 32, 4, padding = 'same', activation = tf.nn.relu, name = 'conv4') 
    pool2 = tf.layers.max_pooling2d(conv4, 2, 2, name = 'pool2')
          
    conv5 = tf.layers.conv2d(pool2, 64, 5, padding = 'same', activation = tf.nn.relu, name = 'conv5')
    conv6 = tf.layers.conv2d(conv5, 64, 5, padding = 'valid', activation = tf.nn.relu, name = 'conv6')
    pool3 = tf.layers.max_pooling2d(conv6, 2, 2, name = 'pool3')

    vec1 = tf.layers.flatten(pool3, name = 'vec1')
    vec2 = tf.layers.dense(vec1, 64, activation = tf.nn.relu, name = 'vec2')
    drop1 = tf.layers.dropout(vec2, rate = select_prob, name = 'drop1')
    vec3 = tf.layers.dense(drop1, 128, activation = tf.nn.relu, name = 'vec3')
    drop2 = tf.layers.dropout(vec3, rate = select_prob, name = 'drop2')
    scores = tf.layers.dense(drop2, 10, activation = None, name = 'scores')
    
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y, logits = scores), name = 'loss')
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    correct = tf.equal(tf.argmax(scores, 1), y)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name = 'accuracy')
    return (x, y, select_prob, optimizer, loss, accuracy)

def draw(accuarcy_training_list, accuarcy_testing_list, loss_training_list, loss_testing_list, epoch_times):
    x1 = np.linspace(1, epoch_times, epoch_times)
    x2 = np.linspace(1, epoch_times, epoch_times)
    # Plot loss figure
    plt.figure()
    plt.plot(x1, loss_training_list, color = 'red', label = 'Training Loss')
    plt.plot(x1, loss_testing_list, color = 'blue', label = 'Testing Loss')
    plt.xlabel('Epoch times')
    plt.ylabel('Loss')
    plt.ylim((0, 1))
    plt.legend()
    # Plot accuracy figure
    plt.figure()
    plt.plot(x2, accuarcy_training_list, color = 'red', label = 'Training Accuracy')
    plt.plot(x2, accuarcy_testing_list, color = 'blue', label = 'Testing Accuracy')
    plt.xlabel('Epoch times')
    plt.ylabel('Accuracy')
    plt.ylim((0, 1.05))
    plt.legend()    
    plt.show()

if __name__ == '__main__':
    training_samples, training_labels, testing_samples, testing_labels, sample_num, row, col, channel = data_import('MNIST.mat')
    
    epoch_times = 20
    batch_size = 60 # 30
    iter_per_epoch = int(sample_num/batch_size)
    learning_rate = 0.001
    loss_training_list = []
    loss_testing_list = []
    accuarcy_training_list = []
    accuarcy_testing_list = []
    
    x, y, select_prob, optimizer, loss, accuracy = Network(row, col, channel, learning_rate)
       
    saver = tf.train.Saver()
    with tf.Session() as sess:    
        k = 0
        max_test_accuracy = 0
        sess.run(tf.global_variables_initializer())
        
        for epoch in range(epoch_times):
            epoch_loss = 0
            epoch_accuracy = 0
            for i in range(iter_per_epoch):
                x_batch = training_samples[i*batch_size:(i+1)*batch_size]
                y_batch = training_labels[i*batch_size:(i+1)*batch_size]
                sess.run(optimizer, feed_dict = {x: x_batch, y: y_batch, select_prob: 0.5})
                
                if i % 100 == 0:
                    k += 1
                    print(k)
                    values = sess.run([loss, accuracy], feed_dict = {x: x_batch, y: y_batch, select_prob: 1.0})
                    loss_training_list.append(values[0])
                    accuarcy_training_list.append(values[1])
                     
                    values = sess.run([loss, accuracy], feed_dict = {x: testing_samples, y: testing_labels, select_prob: 1.0})
                    loss_testing_list.append(values[0])
                    accuarcy_testing_list.append(values[1])
                    print('Testing accuracy (of 10000 samples) =', values[1])
                    
                if values[1] > max_test_accuracy and i % 500 == 0:
                    max_test_accuracy = values[1]
                    saver.save(sess, './data/model.ckpt')
        draw(accuarcy_training_list, accuarcy_testing_list, loss_training_list, loss_testing_list, k)