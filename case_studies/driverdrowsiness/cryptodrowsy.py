# utils
import threading
import pandas as pd
import os
import argparse
from tqdm import tqdm
import time
import pickle
import multiprocessing as mp
import logging
import warnings

# libs
import onnx
import crypten
import crypten.mpc as mpc
import crypten.communicator as comm
from crypten.config import cfg
import torch
import numpy as np
from examples.multiprocess_launcher import MultiProcessLauncher
import torchvision

# model
from .models.crypten_compactcnn import CryptenCompactCNN

def _run_sec_drowsy_model(args:dict):
    '''
    Use the args parameter as a dictionary for holding key argument 
    variables for the yolo runs
    '''
    # import function to run the independent processes
    from .twopc_compactcnn import run_2pc_compactcnn
    
    level = logging.INFO
    if "RANK" in os.environ and os.environ["RANK"] != "0":
        level = logging.CRITICAL
    logging.getLogger().setLevel(level)
    
    # pass all of the arguments
    run_2pc_compactcnn(args)
    
    return 

def multiproc_gpu_drowsy(run_experiment, run_val='0', args:dict=None):
    if args is None: 
        args = {
            "world_size":2,
            "img_size":(640,640), 
            "model": torch.hub.load('ultralytics/yolov5', 'yolov5s', force_reload=True, trust_repo=True),
            "data_path":"source/crypten_source/walkway.pth",
            "run_label":run_val,
            "batch_size":1
            # need to generate the validation data file still
        }
    else: 
        args=args
        
    # the function `run_experiment` ultimately takes the input `args`
    launcher = MultiProcessLauncher(args['world_size'], run_experiment, args, launched=True)
    launcher.start()
    launcher.join()
    launcher.terminate()
    
def main():
    device='cpu'
    
    path = "pretrained/sub9/model.pth"
    model=CryptenCompactCNN()
    model.load_state_dict(torch.load(path, map_location=device))
    model=model.to(device=device)
    
    labels = torch.load("dev_work/test_features_labels/9-crypten_labels.pth")
    compactcnn_args = {
        'world_size':2,
        'img_size':(1,384),
        'model':model,
        'data_path':'dev_work/test_features_labels/9-crypten_features.pth',
        'run_label':'gpu_compactcnn',
        'batch_size':len(labels),
        'folder':'crypten_tmp',
        'device':device, 
        'debug':True,
    }
    multiproc_gpu_drowsy(_run_sec_drowsy_model, f'{device}_val', args=compactcnn_args)
    with open("experiments/crypten_tmp/run_{}.pkl".format('gpu_compactcnn'), 'rb') as f:
        preds, start, end = pickle.load(f)
    with open("experiments/crypten_tmp/comm_tmp_0.pkl", 'rb') as com_0:
        alice_com = pickle.load(com_0)
    with open("experiments/crypten_tmp/comm_tmp_1.pkl", "rb") as com_1:
        bob_com = pickle.load(com_1)
    
    pred_array = np.zeros(len(labels))
    preds = preds.argmax(axis=-1)
    
    print("[INFO]: preds = \n{}".format(preds))
    print("[INFO]: labels = \n{}".format(labels))
    for i in range(len(preds)):
        if preds[i] == labels[i]:
            pred_array[i] = 1
        else: 
            pred_array[i] = 0
    
    print("[INFO]: cost = {}".format(alice_com['bytes'] + bob_com['bytes']))
    cost = (alice_com['bytes'] + bob_com['bytes'])/(2e6) # convert to MB
    round_vals = (alice_com['rounds'] + bob_com['rounds'])/2
    acc = np.sum(pred_array)/len(labels)
    
    print("[RESULT]: Communication cost = {} MB".format(cost/len(labels)))
    print("[RESULT]: Rounds = {}".format(round_vals/len(labels)))
    print("[RESULT]: Accuracy = {}".format(acc))
    print("[RESULT]: run time / inference = {}".format((end-start)/len(labels)))

if __name__ == "__main__":
    main()
    