#exctract weights and save them in numpy format
#save and load torch models
#link to reference help: https://pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html#:~:text=Saving%20and%20Loading%20Model%20Weights%20PyTorch%20models%20store,the%20torch.save%20method%3A%20model%20%3D%20models.vgg16%28pretrained%3DTrue%29%20torch.save%28model.state_dict%28%29%2C%20%27model_weights.pth%27%29

#there is a similar method for saving a trained model in pytorch
import torch
import numpy as np
from compact_cnn import compact_cnn

#load the model
def load_model(file_handle="ELU"): 
    #set the model
    FILE = f'{file_handle}.pth'
    my_net = compact_cnn().double().cuda()
    #load the state dict
    my_net.load_state_dict(torch.load(FILE))
    my_net_dict = my_net.state_dict().items()
    return my_net_dict

#extract the weights - input is my_net_dict
def extract_weights(model_dict, file_handle):
    params = []
    for name, param in model_dict:
        params.append(param.view(-1))
    #concatenate the network parameters
    params = torch.cat(params)
    #convert the tensor into a NumPy array
    model_weights = params.cpu().detach().numpy()
    np.save(f'/home/jjl20011/snap/snapd-desktop-integration/current/Lab/python/pretrained_numpy_models/compactCNN_seed4_subj9.npy', model_weights.astype(np.float64))

def main():
    torch_model_file_handle = input("please input the torch model file path (from wd) excluding the extension: ")
    model_dict = load_model(torch_model_file_handle)
    extract_weights(model_dict, torch_model_file_handle)

if __name__ == "__main__":
    main()
