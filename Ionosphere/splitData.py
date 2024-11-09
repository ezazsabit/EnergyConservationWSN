import numpy as np


def split_data(dataset, label):
    # train-test split
    indices = np.random.permutation(len(dataset))
    training_idx, test_idx = indices[:240], indices[240:]
    x_train = [dataset[i] for i in training_idx]
    x_test = [dataset[i] for i in test_idx]
    y_train = [label[i] for i in training_idx]
    y_test = [label[i] for i in test_idx]

    return x_train, y_train, x_test, y_test