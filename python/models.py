import cPickle as pkl

import tensorflow as tf

import utils

dtype = utils.DTYPE


class LR:
    def __init__(self, input_dim=None, output_dim=1, init_path=None, opt_algo='gd', learning_rate=1e-2,
                 l2_weight=0, random_seed=None):
        init_actions = [('w', [input_dim, output_dim], 'tnormal', dtype),
                        ('b', [output_dim], 'zero', dtype)]
        self.graph = tf.Graph()
        with self.graph.as_default():
            if random_seed is not None:
                tf.set_random_seed(random_seed)
            self.X = tf.sparse_placeholder(dtype)
            self.y = tf.placeholder(dtype)
            self.vars = utils.init_var_map(init_actions, init_path)

            w = self.vars['w']
            b = self.vars['b']
            logits = tf.sparse_tensor_dense_matmul(self.X, w) + b
            self.y_prob = tf.sigmoid(logits)

            self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits, self.y))
            self.loss += l2_weight * tf.nn.l2_loss(w)
            self.optimizer = utils.get_optimizer(opt_algo, learning_rate, self.loss)

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)
            tf.initialize_all_variables().run(session=self.sess)

    def run(self, fetches, feed_dict=None):
        return self.sess.run(fetches, feed_dict)

    def dump(self, model_path):
        var_map = {}
        for name, var in self.vars.iteritems():
            var_map[name] = self.run(var)
        pkl.dump(var_map, open(model_path, 'wb'))
        print 'model dumped at', model_path


class FM:
    def __init__(self, input_dim=None, output_dim=1, factor_order=10, init_path=None, opt_algo='gd', learning_rate=1e-2,
                 l2_w=0, l2_v=0, random_seed=None):
        init_actions = [('w', [input_dim, output_dim], 'tnormal', dtype),
                        ('v', [input_dim, factor_order], 'tnormal', dtype),
                        ('b', [output_dim], 'zero', dtype)]
        self.graph = tf.Graph()
        with self.graph.as_default():
            if random_seed is not None:
                tf.set_random_seed(random_seed)
            self.X = tf.sparse_placeholder(dtype)
            self.y = tf.placeholder(dtype)
            self.vars = utils.init_var_map(init_actions, init_path)

            w = self.vars['w']
            v = self.vars['v']
            b = self.vars['b']
            logits = tf.sparse_tensor_dense_matmul(self.X, w) + b
            X_square = tf.SparseTensor(self.X.indices, tf.square(self.X.values), self.X.shape)
            logits += 0.5 * tf.reshape(tf.reduce_sum(
                tf.square(tf.sparse_tensor_dense_matmul(self.X, v)) - \
                tf.sparse_tensor_dense_matmul(X_square, tf.square(v)),
                1), [-1, output_dim])
            self.y_prob = tf.sigmoid(logits)

            self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits, self.y))
            self.loss += l2_w * tf.nn.l2_loss(w) + l2_v * tf.nn.l2_loss(v)
            self.optimizer = utils.get_optimizer(opt_algo, learning_rate, self.loss)

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)
            tf.initialize_all_variables().run(session=self.sess)

    def run(self, fetches, feed_dict=None):
        return self.sess.run(fetches, feed_dict)

    def dump(self, model_path):
        var_map = {}
        for name, var in self.vars.iteritems():
            var_map[name] = self.run(var)
        pkl.dump(var_map, open(model_path, 'wb'))
        print 'model dumped at', model_path


class PNN1:
    def __init__(self, input_dim=None, output_dim=1, factor_order=10, init_path=None, opt_algo='gd', learning_rate=1e-2,
                 random_seed=None):
        init_actions = []
        self.graph = tf.Graph()
