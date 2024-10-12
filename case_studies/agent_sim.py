'''
Experiment for running multiple agents simultaneusly - each 
system is grabbing a separate video feed from the traffic lights
this is simulated by having a pre-stored set of individual frames, 
and reading the inference one by one in YOLO (for every five frames)

Purpose of experiment is to determine the following metrics
+ communication costs as more threads are running simultaneously
+ secure inference speed as more threads are run simultaneously
+ how do mixed simulations perform (both CompactCNN and YOLO 
  are run simultaneously). This might not work if we try to run 
  the experiments on the GPU since CompactCNN will not work if we 
  run the GPU actively. I wonder if there is a way to isolate the 
  environment variables to that particular thread. 
  
We will obtain separate simulation results - two 
separate servers control these multi-agent processes in 
our setup ... or at least two separate shell environments???
This kind of semantic is easy to argue away later in the paper
I think. Actually mixed CompactCNN / YOLO sims can be run since 
we can do this in the subprocess. The os.environ command needs
to be used in order to accomplish this

HYPOTHESIS: 
As long as the number of processes remains less than 64//3 = 21
then processing should remain optimal. But if we go over this 
amount, then we will encounter performance degradation - maybe this 
would be good to measure n>21
'''

import os
import torch
from driverdrowsiness.cryptodrowsy import multiproc_gpu_drowsy, _run_sec_drowsy_model  # this one runs compactcnn
from driverdrowsiness.models.crypten_compactcnn import CryptenCompactCNN
from traffic_rule_violation.crypten_detect import multiproc_gpu, _run_sec_model        # this one runs yolo

def cryptodrowsy(p_id:int, params:dict):
    '''
    pseudocode: setting the environment variables only works one call (process) above
    1. set up environment for the subprocess
    2. run the secure functionality
    3. record results in designated file
    '''
    if params['device'] == 'cpu':
        os.environ['CUDA_AVAILABLE_DEVICES'] = ''
    tmp_folder = 'crypten_tmp_drowsy_{}'.format(p_id)
    os.mkdir(f'experiments/{tmp_folder}')
    pass
  
def fastsec_yolo(p_id:int, params:dict):
    '''
    pseudocode: 
    1. set up environment for the subprocess
    2. run the secure functionality
    3. record results in designated file
    '''
    if params['device'] == 'cuda':
        os.environ['CUDA_AVAILABLE_DEVICES'] = '0,1'
    tmp_folder = "crypten_tmp_yolo_{}".format(p_id)
    os.mkdir(f'experiments/{tmp_folder}') # make the crypten holding dir
    pass
  
def fetch_sim_vid_data(vid_path):
    pass

def run_agents():
    '''
    pseudocode:
    drowsy_mod = CryptenCompactCNN() # set arch
    drowsy_mod.load_weights # fake func - load weights
    
    rlr_mod = get mod from torch hub # set arch and weights
    
    rlr_params = {
        "world_size":2,
        "img_size":(288,288),
        "model":model, 
        "data_path":#crypten data source,
        "run_label":batch_idx, # just use as a replacable tmp file
        "batch_size":batch_shape,
        "folder":"crypten_tmp",
        "device":device,
        "debug":True
    }
    drows_params = {
        'world_size':2,
        'img_size':(1,384),
        'model':model,
        'data_path':'dev_work/test_features_labels/9-crypten_features.pth',
        'run_label':'gpu_compactcnn',
        'batch_size':len(labels),
        'folder':'crypten_tmp',
        'device':'cpu', 
        'debug':True,
    }

    # each process will need to construct its own crypten_tmp folder so that 
    # they aren't writing over each other's work in memory
    '''
    pass

