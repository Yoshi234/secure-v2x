"""
Preliminary Evaluation of CompactCNN performance (rust model) in Delphi
"""

import argparse
import numpy as np
import os
import random
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from os import path
import scipy.io as sio

SUBJ_NUM = 9
SELECTED_CHANNEL = [28]
#just use the absolute data path to get the data file
ABSOLUTE_PATH = "/home/jjl20011/snap/snapd-desktop-integration/current/Lab/V2V-Delphi-Applications/python/data/dataset.mat"

def generate_eeg_data(data_path):
    """Sample and save a random set of (num_samples) eeg samples"""
    raw_data = sio.loadmat(ABSOLUTE_PATH)

    # x_data contains the actual EEGsample data (all 30 channels by default)
    x_data = np.array(raw_data['EEGsample'])
    # labels is an array (2022, 1) containing the corresponding labels (drowsy or awake)
    # for all 2022 samples in the dataset
    labels = np.array(raw_data['substate'])
    # subidx is an array (2022, 1) containing the corresponding subjnum of each
    # data sample
    subidx = np.array(raw_data['subindex'])

    # re-write x_data to only contain the 28th channel since this is the 
    # only data used for the model
    x_data = x_data[:,SELECTED_CHANNEL,:]

    # filter x_data to only contain items 




