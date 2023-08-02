#exctract weights and save them in numpy format
#save and load torch models
#link to reference help: https://pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html#:~:text=Saving%20and%20Loading%20Model%20Weights%20PyTorch%20models%20store,the%20torch.save%20method%3A%20model%20%3D%20models.vgg16%28pretrained%3DTrue%29%20torch.save%28model.state_dict%28%29%2C%20%27model_weights.pth%27%29

#there is a similar method for saving a trained model in pytorch

import torch
import numpy as np
from models.compact_cnn_pytorch_batch_norm import compact_cnn_pytorch_batch_norm
from models.compact_cnn import compact_cnn

#load the model
def load_model(file_handle="ELU", model_type="compact_cnn"): 
    #set the model
    FILE = f'{file_handle}'
    my_net = None
    if model_type=="compact_cnn":
        my_net = compact_cnn().double().cuda()
    elif model_type=="compact_cnn_pytorch_batch_norm":
        my_net = compact_cnn_pytorch_batch_norm().double().cuda()
    #load the state dict
    my_net.load_state_dict(torch.load(FILE))
    my_net_dict = my_net.state_dict().items()
    return my_net_dict

#extract the weights - input is my_net_dict
def extract_weights(model_dict, file_handle="compact_cnn_pytorch_batch_norm_no_batch.npy"):
    params = []
    for name, param in model_dict:
        if "batch" in name:
            continue
        params.append(param.view(-1))
    #concatenate the network parameters
    params = torch.cat(params)
    #convert the tensor into a NumPy array
    model_weights = params.cpu().detach().numpy()
    #print(sys.path)
    np.save(f'{file_handle}', model_weights.astype(np.float64))

def main():
    torch_model_file_name = "pytorch_batch_norm_tests/pytorch_batch_norm.pth"
    #torch_model_file_handle = input("please input the torch model file path (from wd) excluding the extension: ")
    model_dict = load_model(torch_model_file_name, "compact_cnn_pytorch_batch_norm")
    extract_weights(model_dict)

if __name__ == "__main__":
    main()
