# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 17:41:44 2023

@author: ALARST13
"""

import torch
import scipy.io as sio
import numpy as np
from sklearn.metrics import accuracy_score
import torch.optim as optim
from models.compactcnn import CompactCNN

def test_model(model_path, x_test, y_test):
    # Load pretrained model
    my_net = CompactCNN().double().cuda()
    my_net.load_state_dict(torch.load(model_path))
    my_net.eval()
    
    # test the results
    my_net.train(False)
    with torch.no_grad():
        x_test = torch.DoubleTensor(x_test).cuda()
        answer = my_net(x_test)
        probs = answer.cpu().numpy()
        preds = probs.argmax(axis=-1)
        acc = accuracy_score(y_test, preds)

        print("Accuracy: " + "{:.2f}%".format(acc * 100))

def sequential_testing(model_path, x_test, y_test, res_path):
    my_net = CompactCNN().double().cuda()
    my_net.load_state_dict(torch.load(model_path))
    my_net.eval()

    correct = []
    my_net.train(False)
    with torch.no_grad():
        for i in range(len(y_test)):
            temp_test = torch.DoubleTensor(x_test[i]).reshape(1,1,1,384).cuda()
            answer = my_net(temp_test)
            probs = answer.cpu().numpy()
            preds = probs.argmax(axis=-1)
            correct += [1] if preds == y_test[i] else [0]
            with open(res_path, "a") as f:
                f.write("{} {} {}\n".format(i, y_test[i], preds[0]))


if __name__ == '__main__':
    # load data from the file
    filename = r'data/dataset.mat'
    subjnum = 9
    res_path = "dev_work/experiments/output{}.txt".format(subjnum)

    tmp = sio.loadmat(filename)
    xdata = np.array(tmp['EEGsample'])
    label = np.array(tmp['substate'])
    subIdx = np.array(tmp['subindex'])

    label.astype(int)
    subIdx.astype(int)

    samplenum = label.shape[0]

    # there are 11 subjects in the dataset. Each sample is 3-seconds data from 30 channels with a sampling rate of 128Hz.
    channelnum = 30
    samplelength = 3
    sf = 128

    # ydata contains the label of samples
    ydata = np.zeros(samplenum, dtype=np.longlong)

    for i in range(samplenum):
        ydata[i] = label[i]

    # only channel 28 is used, which corresponds to the Oz channel
    selectedchan = [28]

    # update the xdata and channel number
    xdata = xdata[:, selectedchan, :]
    channelnum = len(selectedchan)
    
    # form the testing data
    testindx = np.where(subIdx == subjnum)[0]
    xtest = xdata[testindx]
    x_test = xtest.reshape(xtest.shape[0], 1, channelnum, samplelength*sf)
    y_test = ydata[testindx]

    model_path = "pretrained/" + "sub" + f"{subjnum}/" + "model.pth"
    test_model(model_path, x_test, y_test)
    sequential_testing(model_path, x_test, y_test, res_path)
