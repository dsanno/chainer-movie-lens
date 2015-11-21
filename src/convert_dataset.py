import argparse
import cPickle as pickle
import numpy as np

parser = argparse.ArgumentParser(description='Convert train and test data')
parser.add_argument('info_file',  type=str, help='information file')
parser.add_argument('train_file', type=str, help='training data file')
parser.add_argument('test_file',  type=str, help='test data file')
parser.add_argument('out_file',   type=str, help='output pkl file')
args = parser.parse_args()

def load_info(file_path):
    with open(file_path) as f:
        user_num, _ = f.readline().split()
        item_num, _ = f.readline().split()
    return (int(user_num), int(item_num))

def load_data(file_path):
    with open(file_path) as f:
        lines = f.readlines()
    data_size = len(lines)
    users = np.zeros((data_size,), dtype=np.int32)
    items = np.zeros((data_size,), dtype=np.int32)
    ratings = np.zeros((data_size, 1), dtype=np.float32)
    timestamps = np.zeros((data_size, 1), dtype=np.float32)
    for i, line in enumerate(lines):
        user, item, rating, timestamp = line.split()
        users[i] = int(user) - 1
        items[i] = int(item) - 1
        ratings[i][0] = float(rating)
        timestamps[i][0] = float(timestamp)
    return (users, items, ratings, timestamps)

print 'Loading information...'
user_num, item_num = load_info(args.info_file)
print 'Loading training data set...'
train_data = load_data(args.train_file)
print 'Loading test data set...'
test_data = load_data(args.test_file)

print 'Saving data set'
with open(args.out_file, 'wb') as f:
    pickle.dump((user_num, item_num, train_data, test_data), f, pickle.HIGHEST_PROTOCOL)
print 'Completed'
