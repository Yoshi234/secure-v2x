# utils
import logging
import os
import time
import pickle

# libs
import crypten
import crypten.communicator as comm
import torch

ALICE=0
BOB=1

def run_2pc_yolo(args:dict):
    '''
    Load the torch hub yolov5 model of interest
    '''
    crypten.init()
    dummy_input = torch.empty(args['batch_size'],3,*args['img_size'])

    # does moving the model to the gpu after encrypting mess it up?
    sec_model = crypten.nn.from_pytorch(args['model'], dummy_input).encrypt(src=ALICE).to(args['device'])
    
    # print("[DEBUG-twopc_yolo.py - 24]: type of sec_model = {}".format(type(sec_model)))
    sec_model.eval()
    
    # encrypt and reshape the data tensor into the proper format
    data_enc = crypten.load_from_party(args['data_path'], src=BOB).reshape(1,3,*args['img_size']).to(args['device'])
    
    # print("[DEBUG-{}]: Running on GPU? = {}".format(comm.get().get_rank(), data_enc.is_cuda))
    start = time.time()
    pred = sec_model(args['device'], data_enc) #device value is required positional argument for my code
    end = time.time()
    
    # decrypt output for the data holder, but not the server
    rank = comm.get().get_rank()
    
    pkl_str = "experiments/gpu_sec_outs/run_{}.pkl".format(args['run_label'])
    pred_dec = pred.get_plain_text(dst=BOB) # only BOB should get the output set "dst=BOB"
    
    if rank == BOB:
        with open(pkl_str, 'wb') as pkl_file:
            pickle.dump([pred_dec, start, end], pkl_file)