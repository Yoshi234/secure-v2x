"""
Evaluate compact_cnn model accuracy on the 
generated test eeg_data of subject 9
"""

from os import path
import scipy.io as sio
import argparse
import numpy as np
import os 
import random 
import sys
from ......python.python_models.compact_cnn_approximation import compact_cnn_approximation
from ......python.python_models.compact_cnn import compact_cnn


# might want to come back to this - is important for generating the 
# "plaintext.npy" file below
def build_model(type):
    """
    Construct model following the given architecture and approx layers
    """
    my_net = None
    if type == "no_approx":
        my_net = compact_cnn().double().cuda()
    elif type == "approx":
        my_net = compact_cnn_approximation().double().cuda()
    
    return my_net
        
def generate_eeg_data(num_samples, dataset, eeg_data_path=None):
    """
    Get the 314 eeg test samples into scope
    """
    # load the dataset into scope
    if eeg_data_path == None: eeg_data_path = os.getcwd()
    
    
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

    if eeg_data_path == None: eeg_data_path = os.getcwd()
    # load image classes
    classes = np.load(path.join(eeg_data_path, "classes.npy"))
    # run inference on all images and track plaintext predictions

    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--weights_path', required=True, type=str,
                        help='<REQUIRED> path to model weights')
    parser.add_argument('type', '--type', type=str, 
                        help='<REQUIRED> type `no_approx` or `approx`')
    # parser.add_argument('-e', '--eeg_data_path', required=True, type=str, 
    #                     help='Path to place images')
    parser.add_argument('-g', '--generate', required=False, type=int, 
                        help='How many eeg_samples to generate (default=314)')
    args = parser.parse_args()

    # load dataset
    eeg_data_path = ""
    dataset = sio.loadmat("../../python/data/dataset.mat")

    # Resolve paths
    weights_path = path.abspath(args.weights_path)
    #eeg_data_path = path.abspath(args.eeg_data_path) 
    #os.makedirs(eeg_data_path, exist_ok=True)

    # Buil model 
    model = build_model(type=args.type)
    
    model = build_model(args.type)

    # pass the loaded dataset as an argument to the data generation function
    if args.generate:
        generate_eeg_data(num_samples=314, dataset=dataset)

    print(f"Accuracy: {test_network(model, eeg_data_path)}")