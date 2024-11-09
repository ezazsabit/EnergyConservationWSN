import torch
import torch.utils.data as utils

def convert2Tensor(trainData, trainLabel, testData, testLabel, batch_size, kwargs):
    # training data
    tensor_x = torch.stack([torch.Tensor(i) for i in trainData])  # transform to torch tensors
    tensor_y = torch.Tensor(trainLabel).long()
    my_dataset = utils.TensorDataset(tensor_x, tensor_y)  # create your datset
    train_loader = utils.DataLoader(my_dataset, batch_size=batch_size, shuffle=True, **kwargs)  # create your dataloader

    # test data
    tensor_x = torch.stack([torch.Tensor(i) for i in testData])  # transform to torch tensors
    tensor_y = torch.Tensor(testLabel).long()
    my_dataset = utils.TensorDataset(tensor_x, tensor_y)  # create your dataset
    test_loader = utils.DataLoader(my_dataset, batch_size=batch_size, shuffle=False, **kwargs)  #
    return train_loader, test_loader