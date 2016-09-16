import cPickle as pkl

import numpy as np
import tensorflow as tf
from scipy.sparse import coo_matrix

DTYPE = tf.float32

FIELD_SIZES = [4, 25, 14, 131227, 43, 364, 5, 765, 996, 2, 2, 4, 2, 4, 2, 5]
INPUT_DIM = 133465
OUTPUT_DIM = 1

NAME_FIELD = {'weekday': 0, 'hour': 1, 'useragent': 2, 'IP': 3, 'region': 4, 'city': 5, 'adexchange': 6, 'domain': 7,
              'slotid': 8, 'slotwidth': 9, 'slotheight': 10, 'slotvisibility': 11, 'slotformat': 12, 'creative': 13,
              'advertiser': 14, 'slotprice': 15}

STDDEV = 1e-3
MINVAL = -1e-2
MAXVAL = 1e-2


def read_data(file_name):
    X = []
    y = []
    with open(file_name) as fin:
        for line in fin:
            fields = line.strip().split()
            y_i = int(fields[0])
            X_i = map(lambda x: int(x.split(':')[0]), fields[1:])
            y.append(y_i)
            X.append(X_i)
    X = np.array(X)
    y = np.reshape(np.array(y), (-1, 1))
    X = libsvm_2_coo(X, (X.shape[0], INPUT_DIM)).tocsr()
    return X, y


def shuffle(data):
    X, y = data
    ind = np.arange(X.shape[0])
    for i in range(7):
        np.random.shuffle(ind)
    return X[ind], y[ind]


def libsvm_2_coo(libsvm_data, shape):
    coo_rows = np.zeros_like(libsvm_data)
    coo_rows = (coo_rows.transpose() + range(libsvm_data.shape[0])).transpose()
    coo_rows = coo_rows.flatten()
    coo_cols = libsvm_data.flatten()
    coo_data = np.ones_like(coo_rows)
    return coo_matrix((coo_data, (coo_rows, coo_cols)), shape=shape)


def csr_2_input(csr_mat):
    coo_mat = csr_mat.tocoo()
    indices = np.vstack((coo_mat.row, coo_mat.col)).transpose()
    values = csr_mat.data
    shape = csr_mat.shape
    return indices, values, shape


def slice(csr_data, start=0, size=-1):
    if size == -1 or start + size >= csr_data[0].shape[0]:
        slc_data = csr_data[0][start:]
        slc_labels = csr_data[1][start:]
    else:
        slc_data = csr_data[0][start:start + size]
        slc_labels = csr_data[1][start:start + size]
    return csr_2_input(slc_data), slc_labels


def init_var_map(init_actions, init_path=None):
    if init_path is not None:
        load_var_map = pkl.load(open(init_path, 'rb'))
        print 'load variable map from', init_path, load_var_map.keys()
    var_map = {}
    for var_name, var_shape, init_method, dtype in init_actions:
        if init_method == 'zero':
            var_map[var_name] = tf.Variable(tf.zeros(var_shape, dtype=dtype), dtype=dtype)
        elif init_method == 'one':
            var_map[var_name] = tf.Variable(tf.ones(var_shape, dtype=dtype), dtype=dtype)
        elif init_method == 'normal':
            var_map[var_name] = tf.Variable(tf.random_normal(var_shape, mean=0.0, stddev=STDDEV, dtype=dtype),
                                            dtype=dtype)
        elif init_method == 'tnormal':
            var_map[var_name] = tf.Variable(tf.truncated_normal(var_shape, mean=0.0, stddev=STDDEV, dtype=dtype),
                                            dtype=dtype)
        elif init_method == 'uniform':
            var_map[var_name] = tf.Variable(tf.random_uniform(var_shape, minval=MINVAL, maxval=MAXVAL, dtype=dtype),
                                            dtype=dtype)
        elif isinstance(init_method, int) or isinstance(init_method, float):
            var_map[var_name] = tf.Variable(tf.ones(var_shape, dtype=dtype) * init_method)
        elif init_method in load_var_map:
            if load_var_map[init_method].shape == tuple(var_shape):
                var_map[var_name] = tf.Variable(load_var_map[init_method])
            else:
                print 'BadParam: init method', init_method, 'shape', var_shape, load_var_map[init_method].shape
        else:
            print 'BadParam: init method', init_method
    return var_map


def activate(weights, activation_function):
    if activation_function == 'sigmoid':
        return tf.nn.sigmoid(weights)
    elif activation_function == 'softmax':
        return tf.nn.softmax(weights)
    elif activation_function == 'relu':
        return tf.nn.relu(weights)
    elif activation_function == 'tanh':
        return tf.nn.tanh(weights)
    elif activation_function == 'elu':
        return tf.nn.elu(weights)
    elif activation_function == 'none':
        return weights
    else:
        return weights


def get_optimizer(opt_algo, learning_rate, loss):
    if opt_algo == 'adaldeta':
        return tf.train.AdadeltaOptimizer(learning_rate).minimize(loss)
    elif opt_algo == 'adagrad':
        return tf.train.AdagradOptimizer(learning_rate).minimize(loss)
    elif opt_algo == 'adam':
        return tf.train.AdamOptimizer(learning_rate).minimize(loss)
    elif opt_algo == 'ftrl':
        return tf.train.FtrlOptimizer(learning_rate).minimize(loss)
    elif opt_algo == 'gd':
        return tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    elif opt_algo == 'padagrad':
        return tf.train.ProximalAdagradOptimizer(learning_rate).minimize(loss)
    elif opt_algo == 'pgd':
        return tf.train.ProximalGradientDescentOptimizer(learning_rate).minimize(loss)
    elif opt_algo == 'rmsprop':
        return tf.train.RMSPropOptimizer(learning_rate).minimize(loss)
    else:
        return tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
