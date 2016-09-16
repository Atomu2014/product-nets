import numpy as np
from sklearn.metrics import roc_auc_score

import utils
from models import LR, FM

train_file = '../data/train.fm.txt'
test_file = '../data/test.fm.txt'
fm_model_file = '../data/fm.model.txt'

input_dim = utils.INPUT_DIM
name_field = utils.NAME_FIELD

train_data = utils.read_data(train_file)
# train_data = utils.shuffle(train_data)
test_data = utils.read_data(test_file)

train_size = train_data[0].shape[0]
test_size = test_data[0].shape[0]
num_feas = len(name_field)

min_round = 1
num_round = 1000
early_stop_round = 40
batch_size = 256

field_sizes = utils.FIELD_SIZES
field_offsets = utils.FIELD_OFFSETS


def train(model):
    history_score = []
    for i in range(num_round):
        fetches = [model.optimizer, model.loss]
        if batch_size > 0:
            ls = []
            for j in range(train_size / batch_size + 1):
                X_i, y_i = utils.slice(train_data, j * batch_size, batch_size)
                feed_dict = {model.X: X_i, model.y: y_i}
                _, l = model.run(fetches, feed_dict)
                ls.append(l)
        elif batch_size == -1:
            X_i, y_i = utils.slice(train_data)
            feed_dict = {model.X: X_i, model.y: y_i}
            _, l = model.run(fetches, feed_dict)
            ls = [l]
        train_preds = model.run(model.y_prob, {model.X: utils.slice(train_data)[0]})
        test_preds = model.run(model.y_prob, {model.X: utils.slice(test_data)[0]})
        train_score = roc_auc_score(train_data[1], train_preds)
        test_score = roc_auc_score(test_data[1], test_preds)
        print '[%d]\tloss:%f\ttrain-auc: %f\teval-auc: %f' % (i, np.mean(ls), train_score, test_score)
        history_score.append(test_score)
        if i > min_round and i > early_stop_round:
            if np.argmax(history_score) == i - early_stop_round and history_score[-1] - history_score[
                        -1 * early_stop_round] < 1e-5:
                print 'early stop\nbest iteration:\n[%d]\teval-auc: %f' % (
                    np.argmax(history_score), np.max(history_score))
                break


algo = 'pnn1'

if algo == 'lr':
    lr_params = {
        'input_dim': input_dim,
        'init_path': None,
        'opt_algo': 'gd',
        'learning_rate': 0.01,
        'l2_weight': 0,
        'random_seed': 0
    }

    model = LR(**lr_params)
elif algo == 'fm':
    fm_params = {
        'input_dim': input_dim,
        'factor_order': 2,
        'init_path': None,
        'opt_algo': 'adam',
        'learning_rate': 0.01,
        'l2_w': 0.01,
        'l2_v': 0.001,
    }

    model = FM(**fm_params)
elif algo == 'pnn1':
    model = None

if algo in {'pnn1'}:
    train_data = utils.split_data(train_data)
    test_data = utils.split_data(test_data)

train(model)

# X_i, y_i = utils.slice(train_data, 0, 100)
# feed_dict = {fm_model.X: X_i, fm_model.y: y_i}
# fetches = [fm_model.tmp1, fm_model.tmp2]
# tmp1, tmp2 = fm_model.run(fetches, feed_dict)
# print tmp1
# print tmp2
# print tmp1.shape
# print tmp2.shape
