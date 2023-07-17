import torch
import numpy as np
from compact_cnn import compact_cnn

def load_model(file_handle="pretrained_torch_models/model_subj_9"):
    FILE = f'{file_handle}.pth'
    my_net = compact_cnn().double().cuda()
    my_net.load_state_dict(torch.load(FILE))
    my_net_dict = my_net.state_dict().items()
    return my_net_dict

if __name__ == '__main__':
    # load the model and write all the parameters to the 
    # printed parameters file
    model_dict = load_model()
    file_mode = "a"
    import os
    if not "parameters.txt" in os.listdir():
        file_mode = "w"

    with open("parameters.txt", file_mode) as f:
        for name, value in model_dict:
            f.write(f"{name:20}:{value}\n")