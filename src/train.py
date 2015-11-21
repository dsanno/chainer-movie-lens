import argparse
import cPickle as pickle
import glob
import numpy as np
import os
import chainer
import chainer.functions as F
from chainer import cuda
from chainer import optimizers
from trainer import Trainer
from model import Model
from movie_lens_model import MovieLensModel

parser = argparse.ArgumentParser(description='Train voice')
parser.add_argument('--gpu',       '-g', default=-1,    type=int, help='GPU ID (negative value indicates CPU)')
parser.add_argument('--input',     '-i', default=None,  type=str, help='input model file path')
parser.add_argument('--output',    '-o', required=True, type=str, help='output model file path')
parser.add_argument('--data_file', '-d', required=True, type=str, help='dataset file path')
parser.add_argument('--iter',            default=200,   type=int, help='number of iteration')
args = parser.parse_args()

gpu_device = None
if args.gpu >= 0:
    cuda.check_cuda_available()
    gpu_device = args.gpu

with open(args.data_file, 'rb') as f:
    (user_num, item_num, train_data, test_data) = pickle.load(f)
if args.input is not None:
    model = Model.load(args.input)
else:
    model = MovieLensModel(user_num, item_num)
optimizer = optimizers.Adam(alpha=0.0001)

def loss_func(y, target):
    return F.mean_squared_error(y, target)

def accuracy_func(y, target):
    return F.mean_squared_error(y, target) ** 0.5

def progress_func(epoch, loss, accuracy, test_loss, test_accuracy):
    print 'epoch: {} done'.format(epoch)
    print('train mean loss={}, accuracy={}'.format(loss, accuracy))
    if test_loss is not None and test_accuracy is not None:
        print('test mean loss={}, accuracy={}'.format(test_loss, test_accuracy))
    if epoch % 10 == 0:
        model.save(args.output)

train_users, train_items, train_ratings, _ = train_data
test_users, test_items, test_ratings, _ = test_data
Trainer.train(model, (train_users, train_items), train_ratings, args.iter, x_test=(test_users, test_items), y_test=test_ratings, batch_size=100, gpu_device=gpu_device, loss_func=loss_func, accuracy_func=accuracy_func, optimizer=optimizer, callback=progress_func)
