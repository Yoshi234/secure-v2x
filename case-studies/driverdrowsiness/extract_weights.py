# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 17:41:44 2023

@author: ALARST13
"""

import torch
import numpy as np
from models.compactcnn import CompactCNN


def extract_weights(model_dict, save_path):
    params = []
    for name, param in model_dict:
        print(name)
        params.append(param.view(-1))
    # Concatenate the network parameters
    params = torch.cat(params)
    # Convert the tensor to a NumPy array
    model_weights = params.cpu().detach().numpy()
    np.save(save_path, model_weights.astype(np.float64))


def main():
    subjnum = 9
    model_path = "pretrained/sub{}/model.pth".format(subjnum)
    save_path = "pretrained/sub{}/model.npy".format(subjnum)

    # Load pretrained model
    my_net = CompactCNN().double().cuda()
    my_net.load_state_dict(torch.load(model_path))
    model_dict = my_net.state_dict().items()

    extract_weights(model_dict, save_path)


if __name__ == '__main__':
    main()
