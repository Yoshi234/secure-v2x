"""
Evaluate compact_cnn model accuracy on the 
generated test eeg_data of subject 9

Must be run from within the main repo folder using the command which follows
python -m delphi.rust.experiments.src.validation.generate_test_eeg_samples no_approx

"""

import pdb
pdb.set_trace()
from os import path
import torch
import scipy.io as sio
import argparse
import numpy as np
import os 
import random 
import sys
from python.python_models.compact_cnn_approximation import compact_cnn_approximation
from python.python_models.compact_cnn import compact_cnn

# debugging function
def print_torch_model_parameters(model):
    for name, value in model.state_dict().items():
        print(f"{name:20}:{value}")
        print(f"{'size':20}:{value.size()}\n")
        print(100*"=")

#  might want to come back to this - is important for generating the 
# "plaintext.npy" file below
def build_model(type):
    """
    Construct model following the given architecture and approx layers
    """
    my_net = None
    if type == "no_approx":
        my_net = compact_cnn().double().cuda()
        my_net.load_state_dict(torch.load("/home/jjl20011/snap/snapd-desktop-integration/current/Lab/V2V-Delphi-Applications/python/pretrained_model_weights/pretrained_torch_models/model_subj_9.pth"))
    elif type == "approx":
        my_net = compact_cnn_approximation().double().cuda()
        my_net.load_state_dict(torch.load("/home/jjl20011/snap/snapd-desktop-integration/current/Lab/V2V-Delphi-Applications/python/pretrained_model_weights/pretrained_torch_models/torch_models_with_poly_approx_relu/model_subj_9_seed0.pth"))

    return my_net
        
def generate_eeg_data(num_samples, dataset, eeg_data_path=None):
    """
    Get the 314 eeg test samples into scope
    """
    # load the dataset into scope
    if eeg_data_path == None: eeg_data_path = "/home/jjl20011/snap/snapd-desktop-integration/current/Lab/V2V-Delphi-Applications/delphi/rust/experiments/src/validation/Eeg_Samples_and_Validation"
    
    xdata = np.array(dataset['EEGsample'])
    label = np.array(dataset['substate'])
    subIdx = np.array(dataset['subindex'])

    label.astype(int)
    subIdx.astype(int)

    samplenum = label.shape[0]

    channelnum = 30
    subjnum = 11
    samplelength = 3
    selectedchan = [28]
    channelnum = len(selectedchan)
    sf = 128

    ydata = np.zeros(samplenum, dtype=np.longlong)
    for i in range(samplenum):
        ydata[i] = label[i]
    xdata = xdata[:,selectedchan,:]

    test_subj = 9
    testindx = np.where(subIdx == test_subj)[0]

    xtest = xdata[testindx]
    #x_test = images shape (314, 1, 1, 384)
    x_test = xtest.reshape(xtest.shape[0], 1, channelnum, samplelength*sf)
    #y_test = classes
    y_test = ydata[testindx]

    for i, eeg_sample in enumerate(x_test):
        np.save(os.path.join(eeg_data_path, f"eeg_sample_{i}.npy"), eeg_sample.flatten().astype(np.float64))
    np.save(path.join(eeg_data_path, f"classes.npy"), y_test.flatten().astype(np.int64))
    # save all eeg_data to .npy format
    
def test_network(model, eeg_data_path=None):
    """Get inference results from the given network
    
    Arguments: 
    - Model --- pass the model object generate by `build_model` function
    to the function 
    - eeg_data_path --- pass the path to which the files should be saved after
    running this function

    Returns: 
    - None"""
    
    print_torch_model_parameters(model)
    if eeg_data_path == None: eeg_data_path = "/home/jjl20011/snap/snapd-desktop-integration/current/Lab/V2V-Delphi-Applications/delphi/rust/experiments/src/validation/Eeg_Samples_and_Validation"
    # load image classes
    classes = np.load(path.join(eeg_data_path, "classes.npy"))
    correct = []

    for i in range(len(classes)):
        # load data and reshape to proper shape
        eeg_sample = np.load(path.join(eeg_data_path, f"eeg_sample_{i}.npy")).reshape(1,1, 1, 384)
        # if i < 5: 
        #     print(f"sample {i}")
        #     print(eeg_sample)
       
        with torch.no_grad():
        # run inference on all images and track plaintext predictions
            temp_test = torch.DoubleTensor(eeg_sample).cuda()
            answer = model(temp_test)
            probs = answer.cpu().numpy()
            preds = probs.argmax(axis=-1)[0]
            #print(f"{preds} -> {classes[i]}")
            correct += [1] if preds == classes[i] else [0]
    np.save(path.join(eeg_data_path, "plaintext.npy"), np.array(correct))
    return 100*(sum(correct) / len(classes))

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-w', '--weights_path', required=True, type=str,
    #                     help='<REQUIRED> path to model weights')
    # parser.add_argument('type', '--type', type=str, 
    #                     help='<REQUIRED> type `no_approx` or `approx`')
    # parser.add_argument('-e', '--eeg_data_path', required=True, type=str, 
    #                     help='Path to place images')
    # parser.add_argument('-g', '--generate', required=False, type=int, 
    #                     help='How many eeg_samples to generate (default=314)')
    # args = parser.parse_args()

    # load dataset -- figure out which location to put the files in later
    eeg_data_path = "/home/jjl20011/snap/snapd-desktop-integration/current/Lab/V2V-Delphi-Applications/delphi/rust/experiments/src/validation/eeg_test_samples_subject9"
    dataset = sio.loadmat("/home/jjl20011/snap/snapd-desktop-integration/current/Lab/V2V-Delphi-Applications/python/data/dataset.mat")

    # Resolve paths
    # weights_path = path.abspath(args.weights_path)
    #eeg_data_path = path.abspath(args.eeg_data_path) 
    #os.makedirs(eeg_data_path, exist_ok=True)
    if len(sys.argv) < 2:
        print("Usage: {sys.agrv[0]} model_type")
        print("you can choose model_type=approx or no_approx")
        exit()

    type = sys.argv[1]

    # Build model 
    model = build_model(type=type)

    # pass the loaded dataset as an argument to the data generation function
    generate_eeg_data(num_samples=314, dataset=dataset)

    print(f"Accuracy: {test_network(model)}")