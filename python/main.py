import numpy as np
from sklearn.metrics import roc_auc_score

import utils
from models import LR, FM, PNN1, PNN2

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
early_stop_round = 50
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
                _, l = model.run(fetches, X_i, y_i)
                ls.append(l)
        elif batch_size == -1:
            X_i, y_i = utils.slice(train_data)
            _, l = model.run(fetches, X_i, y_i)
            ls = [l]
        train_preds = model.run(model.y_prob, utils.slice(train_data)[0])
        test_preds = model.run(model.y_prob, utils.slice(test_data)[0])
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


algo = 'pnn2'

if algo == 'lr':
    lr_params = {
        'input_dim': input_dim,
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
        'opt_algo': 'adam',
        'learning_rate': 0.01,
        'l2_w': 0.01,
        'l2_v': 0.001,
    }

    model = FM(**fm_params)
elif algo == 'pnn1':
    pnn1_params = {
        'layer_sizes': [field_sizes, 10, 1],
        'layer_acts': [None, None, None],
        'layer_keeps': [1, 1, 1],
        'opt_algo': 'gd',
        'learning_rate': 1,
        'layer_l2': [0.001, 0.001, 0.001],
        'kernel_l2': 0.001,
        'random_seed': 0
    }

    model = PNN1(**pnn1_params)
elif algo == 'pnn2':
    pnn2_params = {
        'layer_sizes': [field_sizes, 10, 1],
        'layer_acts': [None, None, None],
        'layer_keeps': [1, 1, 1],
        'opt_algo': 'gd',
        'learning_rate': 1,
        'layer_l2': [0.001, 0.001, 0.001],
        'kernel_l2': 0.001,
        'random_seed': 0
    }

    model = PNN2(**pnn2_params)

if algo in {'pnn1', 'pnn2'}:
    train_data = utils.split_data(train_data)
    test_data = utils.split_data(test_data)

train(model)

# X_i, y_i = utils.slice(train_data, 0, 100)
# fetches = [model.tmp1, model.tmp2]
# tmp1, tmp2 = model.run(fetches, X_i, y_i)
# print tmp1.shape
# print tmp2.shape
