# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 23:50:11 2020

@author: strik
"""


import tensorflow.compat.v1 as tf
from tensorflow.contrib import rnn
import numpy as np
from numpy import genfromtxt
# from sklearn import datasets
# from sklearn.cross_validation import train_test_split
# import sklearn

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
# from sklearn.metrics import precision_score
# from sklearn.metrics import recall_score, f1_score
from sklearn.preprocessing import StandardScaler



import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------------


#from tensorflow.contrib import rnn

# from tensorflow.python.ops import rnn, rnn_cell

# -------------------------------------------------------------------------------


#linux paths
# path = r'/home/mitch/Desktop'
# path = r'/home/mitch/Desktop/research/ITS-490-master'

#windows path
# path = r'C:\Users\strik\Desktop\490-ubuntu\490'
# path2 = r'C:\Users\strik\Desktop\490-ubuntu\490\data\countries'
# path = r'C:\Users\strik\Desktop\ITS-490'

#wsl path
# path = r'/mnt/c/Users/strik/Desktop/490-ubuntu/490'

# df = pd.read_csv(os.path.join(path, 'Redrunner-rnn.csv'), usecols=['event-id', 'visible', 'timestamp', 
#                                                         'location-long', 'location-lat', 'tag-local-identifier', 'individual-local-identifier', 
#                                                         'eobs:temperature', 'height-above-ellipsoid', 'heading'])



# -------------------------------------------------------------------------------

#parameters for the main loop

# learning_rate = 0.001
# n_epochs = 100000  ##27000  
# batch_size = 100


# #to predict the class per 28x28 image, we now think of the image
# #as a sequence of rows. Therefore, you have 28 rows of 28 pixels
# #each

# chunk_size = 316898 # MNIST data input (img shape: 28*28)
# n_chunks = 316898 # chunks per image
# rnn_size = 128 # size of rnn
# n_classes = 12 # MNIST total classes (0-9 digits) ## B

tf.reset_default_graph()
learning_rate = 0.1
n_epochs = 10000
batch_size = 10000
display_step = 50

# Network Parameters
chunk_size = 3  #num_input
n_chunks = 1000 # timestamp
num_hidden = 128 # hidden layer num of features
num_classes = 12 #  total classes 


# -------------------------------------------------------------------------------


# load MNIST data
# x_train = genfromtxt('training-file-rnn.csv', delimiter=',',usecols=(i for i in range(1,6)))
x_train = genfromtxt('training-file-rnn-labeled.csv', delimiter=',',usecols=(0,2,3,), skip_header=0, dtype='<U130')
y_train = genfromtxt('training-file-rnn-labeled.csv', delimiter=',', usecols=(5), skip_header=0, dtype='<U130')

x_test = genfromtxt('testing-file-rnn-labeled.csv', delimiter=',', usecols=(0,2,3,) , skip_header=0, dtype='<U130' )
y_test = genfromtxt('testing-file-rnn-labeled.csv', delimiter=',', usecols=(5), skip_header=0, dtype='<U130')


# x_train = x_train[ int(x_train.shape[0]/4): , : ]
# x_test = x_test[ int(x_test.shape[0]/4):  ,   :]
# y_train = y_train[ int(y_train.shape[0]/4):  ]
# y_test = y_test[ int(y_test.shape[0]/4):  ]

x_train = x_train[ :n_chunks , : ]
x_test = x_test[ :n_chunks  ,   :]
y_train = y_train[ :n_chunks  ]
y_test = y_test[ :n_chunks  ]


sc = StandardScaler()
sc.fit(x_train)

x_train_normalized = sc.transform(x_train)
x_test_normalized  = sc.transform(x_test)


def convertOneHot_data2(data):
    y=np.array([int(i) for i in data])
    #print y[:20]
    rows = len(y)
    columns = y.max() + 1
    # columns = y.max()
    a = np.zeros(shape=(rows,columns))
    print(a.shape)
    #print a[:20,:]
    print('ROWS  ', rows)
    print('COLUMNS  ', columns)
    #rr = raw_input()
    # y_onehot=[0]*len(y)
    for i,j in enumerate(y):
        # y_onehot[i]=np.array([0]*(y.max() + 1) )
        # y_onehot[i][j]=1
        
        # print(a[i][j])
        a[i][j]=1
    return (a)

# y_train_onehot = convertOneHot_data2(y_train)

# ---------------------------------------------------------------------------------
# one-hot encoding

depth = 12
# y_train = tf.Variable(y_train)

# y_train_onehot = tf.one_hot(y_train, depth)
# print('-------------------------HERE--------------------------')
# y_test_onehot  = tf.one_hot(y_test, depth)

y_train_onehot = convertOneHot_data2(y_train)
print('-------------------------HERE--------------------------')
y_test_onehot  = convertOneHot_data2(y_test)




# -------------------------------------------------------------------------------

## print stats 
precision_scores_list = []
accuracy_scores_list = []

def print_stats_metrics(y_test, y_pred):    
    print('Accuracy: %.2f' % accuracy_score(y_test,   y_pred) )
    print('----------------------fds_-----------')
    
    accuracy_scores_list.append(accuracy_score(y_test,   y_pred) )
    confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
    
    print("confusion matrix")
    print(confmat)
    print(pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True))
    #precision_scores_list.append(precision_score(y_true=y_test, y_pred=y_pred))
    #print('Precision: %.3f' % precision_score(y_true=y_test, y_pred=y_pred))
    #print('Recall: %.3f' % recall_score(y_true=y_test, y_pred=y_pred))
    #print('F1-measure: %.3f' % f1_score(y_true=y_test, y_pred=y_pred))

# -------------------------------------------------------------------------------


def plot_metric_per_epoch():
    x_epochs = []
    y_epochs = [] 
    for i, val in enumerate(accuracy_scores_list):
        x_epochs.append(i)
        y_epochs.append(val)
    
    plt.scatter(x_epochs, y_epochs,s=50,c='lightgreen', marker='s', label='score')
    plt.xlabel('epoch')
    plt.ylabel('score')
    plt.title('Score per epoch')
    plt.legend()
    plt.grid()
    plt.show()


# -------------------------------------------------------------------------------


def RNN(x):
    W = tf.Variable(tf.random_normal([num_hidden, num_classes]))
    b = tf.Variable(tf.random_normal([num_classes]))

    print(x.shape)
    x = tf.transpose(x, [1, 0, 2]) 
    x = tf.reshape(x, [-1, chunk_size] )
    x = tf.split(x, n_chunks, 0)
    
    # Unstack to get a list of 'n_chunks' tensors of shape (batch_size, n_input)
    # x = tf.unstack(x, n_chunks, 0)

    # print(x)

    # lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0, activation='relu' )
    # lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0 )
    # lstm_cell = rnn.LSTMCell(num_hidden, forget_bias=1.0 )
    # lstm_cell = rnn.RNNCell(num_hidden )
    # lstm_cell = rnn.GRUCell(num_hidden,  )
    lstm_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(num_hidden),rnn.BasicLSTMCell(num_hidden)])

    print('--------------------\n',lstm_cell,'\n------------------------------')    

    # outputs, states = rnn.rnn(lstm_cell, x, dtype=tf.float32)
    print('\nHERE\n')
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
    # outputs, states = rnn.static_bidirectional_rnn(lstm_cell, lstm_cell, x, dtype=tf.float32)


    print('\nHERE 2\n')

    rnn_output =  tf.matmul(outputs[-1], W) + b
    
    print('\nHERE 3\n')

    return rnn_output

# -------------------------------------------------------------------------------


def loss_deep_rnn(output, y_tf):
    print('\nHERE 4\n')
    xentropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=output, labels=y_tf)
    # xentropy = tf.nn.tanh(output)
    loss = tf.reduce_mean(xentropy)
    return loss


# -------------------------------------------------------------------------------


def training(cost):
    print('\nHERE 5\n')
    # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(cost)
    return train_op

# -------------------------------------------------------------------------------


def evaluate(output, y_tf):
    correct_prediction = tf.equal(tf.argmax(output,1), tf.argmax(y_tf,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy


# -------------------------------------------------------------------------------


# x_tf = tf.placeholder("float", [None, n_chunks, chunk_size ])
x_tf = tf.placeholder("float", [None, n_chunks, chunk_size ])
# x_tf = tf.reshape(x_tf, [-1, n_chunks, chunk_size])

y_tf = tf.placeholder("float", [None, num_classes])

# -------------------------------------------------------------------------------

         
output = RNN(x_tf) #logits
cost = loss_deep_rnn(output, y_tf)
train_op = training(cost)
eval_op = evaluate(output, y_tf)

# -------------------------------------------------------------------------------

## for metrics

y_p_metrics = tf.argmax(output, 1)

# -------------------------------------------------------------------------------

# Initialize and run

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

# -------------------------------------------------------------------------------


step = 1
# Keep training until reach max iterations
for i in range(n_epochs):
    for batch_n in range(batch_size):
        # batch_x = x_train[:,:,np.newaxis]
        batch_x = np.reshape(x_train_normalized, (1,) + x_train_normalized.shape)
        # batch_x = batch_x.transpose(batch_x, (2,0,1))
        batch_y = y_train_onehot
        sta = batch_n*batch_size
        end = sta = batch_size
    
        # print('\nHERE 6\n')
        sess.run(train_op, feed_dict={x_tf:batch_x, y_tf: batch_y})
        # print('\nHERE 7\n')
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([cost, eval_op], feed_dict={x_tf: batch_x,
                                                                 y_tf: batch_y})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))
                
    
            test_len = 1000
            # test_data = x_test.reshape((-1, n_chunks, chunk_size))
            test_data = np.reshape(x_test_normalized, (1,) + x_test_normalized.shape)

            test_label = y_test_onehot[:test_len]
            # test_label = np.reshape(test_label, (test_len,12))
            # test_label = y_test
            # print("\t\t\t\t\t\t\t Testing Accuracy:", \
            #      y_result_metrics = sess.run(eval_op, feed_dict={x_tf: test_data, y_tf: test_label}))
            y_result_metrics = sess.run(eval_op, feed_dict={x_tf: test_data, y_tf: test_label})
            print("\t\t\t\t\t\t\t Testing Accuracy: ", y_result_metrics)    
            
            
            y_true = np.argmax(y_test_onehot,1)
            # y_result_metrics = y_result_metrics
            # print_stats_metrics(y_true, y_result_metrics)

        step += 1
    
        # loss, acc = sess.run([cost, eval_op], feed_dict={x_tf: batch_x[sta:end,:], y_tf: batch_y[sta:end,:]})
    
        # result = sess.run(eval_op, feed_dict={x_tf: x_test[sta:end,:], y_tf: y_test_onehot[sta:end,:]})
    
        # result2,  = sess.run([eval_op, ], feed_dict={x_tf: x_test[sta:end,:], y_tf: y_test_onehot[sta:end,:]})
    
        # print("training {},{}".format(step,result))
        # print("testing {},{}".format(step,result2))
        
        # # y_true = np.argmax(y_test_onehot,1)
        
        # # print('here')
        
        # # print_stats_metrics(y_true, y_result_metrics)
            # step = step + 1


# -------------------------------------------------------------------------------


print('---------------done----------------')














