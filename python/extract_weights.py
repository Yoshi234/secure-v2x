#exctract weights and save them in numpy format
#save and load torch models
#link to reference help: https://pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html#:~:text=Saving%20and%20Loading%20Model%20Weights%20PyTorch%20models%20store,the%20torch.save%20method%3A%20model%20%3D%20models.vgg16%28pretrained%3DTrue%29%20torch.save%28model.state_dict%28%29%2C%20%27model_weights.pth%27%29

#there is a similar method for saving a trained model in pytorch
import os
import sys
import torch
import numpy as np
from models.compact_cnn_pytorch_batch_norm import compact_cnn_pytorch_batch_norm
from models.compact_cnn import compact_cnn

#load the model
def load_model(file_handle="ELU", model_type="compact_cnn"): 
    #set the model
    FILE = f'{file_handle}.pth'
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
    file_handle = file_handle.split("/")
    #set the file_handle as the last index of the list created
    file_handle = file_handle[-1]
    params = []
    for name, param in model_dict:
        if "batch" in name:
            continue
        #flatten the tensor
        params.append(param.view(-1))
    #concatenate the network parameters
    params = torch.cat(params)
    #convert the tensor into a NumPy array
    model_weights = params.cpu().detach().numpy()
    #print(sys.path)
    os.chdir('pretrained_numpy_models')
    np.save(f'{file_handle}', model_weights.astype(np.float64))

def updated_extract_weights(model_dict, file_handle):
    '''
    Updated functionality for extracting weights from the saved pytorch model
    
    Arguments:
    - model_dict --- pytorch `state_dict` object containing all of the parameters
    which were saved for the model
    - file_handle --- the name of the file to save the numpy weights to
    
    Returns: 
    - None
    
    Important features:
    Although most of this function is the same as that from `extract_weights`,
    this function compresses the batch normalization and convolutional layer parameters
    into a single set of parameters, called `conv_fold` and `bias_fold`. These parameters
    are defined as follows
    - `conv_fold = gamma1 * conv.weight`
    - `bias_fold = gamma2 * conv.bias + beta`
    '''
    file_handle = file_handle.split("/")
    #set the file_handle as the last index of the list created
    file_handle = file_handle[-1]
    interest_parameters = {"conv.weight":None, "conv.bias":None, "batch.gamma":None, "batch.beta":None}
    params = []
    #define a new pair of parameters
    for name, param in model_dict:
        #get params of interest
        if name in interest_parameters:
        #conv.weight
        #conv.bias
            interest_parameters[name] = param
    #define the parameter's value
    #review the parameters of the gamma matrix
    bias = interest_parameters["conv.bias"]
    beta = interest_parameters["batch.beta"]
    gamma2 = interest_parameters["batch.gamma"]
    gamma1 = interest_parameters["batch.gamma"]
    conv = interest_parameters["conv.weight"]
    #reshape the gamma matrix to be compatible with convolution weights
    gamma1 = interest_parameters["batch.gamma"].view([32, 1, 1, 1])
    gamma1 = gamma1.expand(int(conv.size(0)), int(conv.size(1)), int(conv.size(2)), int(conv.size(3)))
    
    gamma2 = gamma2.view([32])
    beta = beta.view([32])
    
    #set, flatten, and append the parameters to parameter list
    conv_fold = gamma1 * conv
    params.append(conv_fold.view(-1))
    bias_fold = gamma2 * bias + beta
    params.append(bias_fold.view(-1))
    
    for name, param in model_dict:
        if name not in interest_parameters:
            params.append(param.view(-1))
            
    params = torch.cat(params)
    
    model_weights = params.cpu().detach().numpy()
    #print(sys.path)
    #save the trained numpy models in pretrained numpy models
    os.chdir('modified_numpy_models')
    np.save(f'{file_handle}', model_weights.astype(np.float64))
        
def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]}  torch_state_dict  extraction_method")
        print("\t exclude the extension from the torch state dict file name")
        print("\t extraction_method: {'extract_weights', 'updated_extract_weights'}")
        exit()
    torch_model_file_name = sys.argv[1]
    function = sys.argv[2]
    model_dict = load_model(torch_model_file_name, "compact_cnn")
    
    if function == "extract_weights": extract_weights(model_dict, torch_model_file_name)
    #torch_model_file_handle = input("please input the torch model file path (from wd) excluding the extension: ")
    else: updated_extract_weights(model_dict, torch_model_file_name)

if __name__ == "__main__":
    main()
