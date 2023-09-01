import sys

import torch
import numpy as np
from models.compact_cnn import compact_cnn
from models.compact_cnn_pytorch_batch_norm import compact_cnn_pytorch_batch_norm


def load_model(file_handle="pretrained_torch_models/model_subj_9", model="compact_cnn"):
    FILE = f'{file_handle}.pth'
    my_net = None
    if model == "compact_cnn":
        my_net = compact_cnn().double().cuda()
    elif model == "compact_cnn_pytorch_batch_norm":
        my_net = compact_cnn_pytorch_batch_norm().double().cuda()
    my_net.load_state_dict(torch.load(FILE))
    my_net_dict = my_net.state_dict().items()
    return my_net_dict

if __name__ == '__main__':
    # load the model and write all the parameters to the 
    # printed parameters file
    results_file = input("Please enter the name of the file to write to")
    model_dict = load_model("pytorch_batch_norm_tests/pytorch_batch_norm", model="compact_cnn_pytorch_batch_norm")
    file_mode = "w"
    import os
    #if not results_file in os.listdir():
    #    file_mode = "w"

    with open(results_file, file_mode) as f:
        for name, value in model_dict:
            if "batch" in name:
                continue
            # f.write(f"{name:20}:{value}\n")
            # f.write(f"{'size':20}:{value.size()}\n\n")
            # f.write(100*"="+"\n")
            print(name)
