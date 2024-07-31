'''
Author: Yoshi234 7/25/24

Validation of YOLOv5 models on the COCO validation data set
Follows from the yolov5/val.py validation script. See 
https://github.com/ultralytics/yolov5/blob/master/val.py 
Inference time implementation of Secure-RLR utilizes CrypTen
running on a single GPU. Custom implementation of modules is 
required in order to process YOLO in CrypTen. 
'''

# utils
import pandas as pd
import os
import argparse
from utils.torch_utils import select_device
from utils.general import (
    cv2, non_max_suppression, scale_boxes, check_dataset, xywh2xyxy
)
from utils.augmentations import letterbox
from utils.metrics import box_iou, ap_per_class
from tqdm import tqdm
import time
import pickle
import multiprocessing as mp
from ultralytics.utils.plotting import Annotator, colors, save_one_box
import logging

# libs
import onnx
import crypten
import crypten.mpc as mpc
import crypten.communicator as comm
import torch
from models.common import DetectMultiBackend, AutoShape
import numpy as np
from examples.multiprocess_launcher import MultiProcessLauncher

# scripts
from crypten_detect import multiproc_gpu, _run_sec_model

class img_info:
    def __init__(self, data, batch_idx, sub_idx, file, label, og_size, new_size):
        self.dat = data             # stores idx in detection list
        self.file = file            # stores img file (str)
        self.label = label          # stores label file (str)
        self.batch_idx = batch_idx  # stores batch index (the index of the batch which )
        self.sub_idx = sub_idx      # stores index of element in the given batch
        self.og_size = og_size
        self.new_size = new_size

# globals
SUPPORTED_IMG_TYPES = {"png", "jpg", "jpeg"}

def read_label_file(file_path, rearrange=False):
    '''
    reads a coco .txt file and returns a tensor of bounding boxes/class labels
    corresponding to the output
    '''
    if file_path.split(".")[-1] != "txt": 
        raise ValueError("[ERROR-crypten_val 44]: only .txt label files can be read")
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
        # print("[INFO]: initial lines - {}".format(lines))
        lines = [line.split(" ") for line in lines]
        for i in range(len(lines)):
            for j in range(len(lines[i])):
                lines[i][j] = float(lines[i][j])
        lines = [torch.tensor(line) for line in lines]
        lines = torch.stack(lines)
    
    for i in range(len(lines)):
        lines[i][1:] = xywh2xyxy(lines[i][1:]) # this tensor can be of whatever shape you want
    
    return lines[:,[1,2,3,4,0]] if rearrange else lines # rearrange the output columns if necessary

def read_img_file(file_path, img_size=None, stride=32):
    '''
    reads in an image file, and returns the corresponding torch tensor
    and shape values for the original image and numpy image
    '''
    im0 = cv2.imread(file_path)
    # print("[DEBUG-82 - read_img_file] im0.shape = {}".format(im0.shape))
        
    if img_size is None:
        w = im0.shape[1]
        h = im0.shape[0]
        
        if w % stride  == 0: w = w # format width
        elif w % stride != 0: w = (w//stride) * stride
        
        if h % stride == 0: h = h # format height
        elif h % stride != 0: h = (h//stride) * stride
        
    img_size = (h,w)
    
    im = letterbox(im0, img_size, stride=stride, auto=False, scaleFill=True)[0]
    im = im.transpose((2,0,1))[::-1]
    im = np.ascontiguousarray(im)
    im_tensor = torch.from_numpy(im)/255
    # print("[DEBUG-90 - read_img_file] im0.shape = {}".format(im_tensor.shape))
    
    return im_tensor, tuple(im0.shape[-3:-1]), img_size # return the tensor to load and process

def load_and_process(img_folder, lbs_folder, load_size=None, start=0):
    '''
    loads and processes data from the specified dataset, creating 
    a single pytorch file, or set of files to hold a batched 
    set of numpy image arrays
    
    - load_size and start specify index values for loading a specific set
      of images from the folder structure so that not all images are loaded
      into RAM at the exact same time. This is important for much larger 
      batches of data. 
    '''
    if load_size is None: 
        load_size=len(os.listdir(lbs_folder))

    lbl_dir = os.listdir(lbs_folder)
    img_dir = os.listdir(img_folder)
    
    labels = [] # list of tensors 
    for i in range(start,len(lbl_dir)): 
        labels.append(read_label_file("{}/{}".format(lbs_folder, lbl_dir[i])))
        
    imgs = [] # list - tensor of images, separate into batches
    for j in range(start,len(img_dir)):
        imgs.append(read_img_file("{}/{}".format(img_folder, img_dir[j])))
        
    imgs = torch.stack(imgs) # convert tensor list into a single tensor    
    
    return imgs, labels # labels correspond index wise to the tensors
    
def sec_pred():
    # TODO: implement secure prediction (follow crypten detect)
    pass
    
def validate_crypten_yolo(
    data,
    pln_src:str="source/plaintext_validation",
    model_type:str="yolov5s",
    device:str='',
    batch_size:int=32,
    img_size:tuple=(640,640),
    conf_thres:float=0.001,
    iou_thres:float=0.6, 
    max_det:int=300,
):
    '''
    Evaluates (1) Crypt-YOLOv5 and (2) YOLOv5 models on the COCO 
    validation dataset. 
    
    Args:
    - data --- str: path to a yaml file or a dataset dictionary
    
    Returns: 
    - dict: precision, recall, mAP50, and mAP50-95
    '''
    model = torch.hub.load('ultralytics/yolov5', model_type, force_reload=True, trust_repo=True)
    model = model.to(device) # move model to specified device
    model.eval() # set eval mode for plaintext model
    
    data = check_dataset(data)
    
    iouv = torch.linspace(0.5, 0.95, 10).to(device)
    niou = iouv.numel()
    
    model.warmup(imgsz=(1,3,*img_size))
    task = 'val'    
    
def plaintext_val(
    batch_size=32,
    conf_thres=0.25, 
    iou_thres=0.45, 
    max_det=1000, 
    classes=None,
    agnostic_nms=None,
    lbs_folder="/mnt/data/coco128/coco128/labels/train2017",
    imgs_folder="/mnt/data/coco128/coco128/images/train2017",
    model='yolov5s',
    device='cpu',
    plain=True,
    results_name="plaintext_val"
):
    '''
    NOTE: although the authors of CrypTen claim GPU support is 
    provided, there is some computation over which we can't compute
    computations on unencrypted module parameters (not trained weights)
    and encrypted crypten cryptensors - this is an issue on the GPU 
    in particular. When computing on the CPU, this issue does not appear
    for some reason. 
    
    Takes as input the plaintext model, and runs validation on the 
    COCO dataset for specified batch sizes. Ideally, all of the images in the dataset
    should be split up into multiple folders, and batches of eval. be run on each 
    folder of the dataset.
    
    NOTE: the aspect ratio of images NEEDS to be maintained. Adjusting the 
    aspect ratio of the image can have drastic consequences for the output
    bounding box detections. So, we need to load in all the images, and group
    them by their aspect ratios. If they are close enough, they can be 
    computed together. Otherwise, they need to be placed in a separate 
    compute batch. 
    '''
    # load model
    model = torch.hub.load('ultralytics/yolov5', model, force_reload=True, trust_repo=True)
    mod_names, mod_stride = model.names, model.stride
    model = model.to(device) # move model to correct device
    
    lbs_files = os.listdir(lbs_folder) # set file names
    imgs_files = os.listdir(imgs_folder)
    
    labels = [] # list of label tensors
    
    im_batches = [] # set empty list of image batches to run
    cur_batches = dict() # reset every time you are finished filling a batch
    
    # compute prediction and stats for each image
    imgs_info = dict() # init empty img info dictionary
    
    i=0
    for im in imgs_files:
        im_name = im.split(".")[0]
        lb = None
        for lb_name in lbs_files: # search for matching label
            if im_name in lb_name: 
                lb = lb_name
                break
        if lb is None: continue # go to next image if no corr. label
        
        # read and append label to list
        labels.append(read_label_file("{}/{}".format(lbs_folder,lb))) 
        im_tensor, og_size, new_size = read_img_file(f"{imgs_folder}/{im}", stride=mod_stride)
        
        # this method doesn't preserve order, we need some kind of other identifier   
        if len(cur_batches['']) < batch_size:             
            cur_batches[new_size].append(im_tensor)
        else:
            cur_batches[new_size] = [] # reset current batch list
            cur_batches[new_size].append(im_tensor)
            
        imgs_info[i] = (og_size, new_size) # tuple of size tuples
        i+=1 # increment counter
      
    if len(cur_batches[new_size]) < batch_size:       
        im_batches.append(cur_batches[new_size]) # always append last batch (if not already appended)
    
    # convert batches into tensors
    for i in range(len(im_batches)): im_batches[i] = torch.stack(im_batches[i])
    
    detections = [] # list of prediction tensors
    
    # run inference for each batch
    for batch in im_batches:
        if plain:
            start = time.time()
            preds = model(batch.to(device)) # make sure batch is on correct device
            end = time.time()
            
        # TODO: finish implementation for secure validation too
        # elif not plain:
        #     _ = sec_pred(plaintext_model=model, input=batch)
        #     with open("", "rb") as f: # read results from pickle file
        #         preds, start, end = pickle.load(f)
        
        preds = non_max_suppression(
            prediction=preds,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            classes=classes,
            agnostic=agnostic_nms, 
            max_det=max_det
        )
        
        for pred in preds: # append each prediction to a single detections list
            detections.append(pred)
            
    metrics = []
    vals = []
    stats = []
    
    iouv = torch.linspace(0.5, 0.95, 10, device=device)
    
    assert len(detections) == len(labels), "[ERROR-264]: labels / detections don't match"
    
    for i in range(len(detections)):
        lbl = labels[i]
        det = detections[i]
        og_size, new_size = imgs_info[i]
        
        correct = np.zeros((det.shape[0], iouv.shape[0])).astype(bool)
        correct_class = lbl[:,0:1] == det[:,5]
        
        det[:,:4] = scale_boxes(new_size, det[:,:4], og_size)
        lbl[:,1:] *= torch.tensor((og_size[1], og_size[0], og_size[1], og_size[0]))
        
        iou_score = box_iou(lbl[:,1:], det[:,:4])
        
        for j in range(len(iouv)):
            x = torch.where((iou_score > iouv[i]) & correct_class)
            if x[0].shape[0]:
                matches = torch.cat((torch.stack(x,1), iou_score[x[0], x[1]][:,None]), 1).cpu().numpy()
                if x[0].shape[0] > 1:
                    matches = matches[matches[:,2].argsort()[::-1]]
                    matches = matches[np.unique(matches[:,1], return_index=True)[1]]
                    matches = matches[np.unique(matches[:,0], return_index=True)[1]]
                correct[matches[:,1].astype(int), i] = True
        correct = torch.tensor(correct, dtype=torch.bool, device=iouv.device)
        
        # append results to stats list tuples=(correct, bounding box, pred class, true class)
        stats.append((correct, det[:,:4], det[:,5], lbl[:,0]))
        
    # format the stats list and save results
    stats = [torch.cat(x,0).cpu().numpy() for x in zip(*stats)]
    tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, plot=False, save_dir="experiments/batched_val", names=mod_names)
    ap50, ap = ap[:,0], ap.mean(1)
    mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
    
    metric_res = {
        "true positive": tp, 
        "false positive": fp, 
        "precision": p, 
        "recall": r,
        "f1 score": f1,
        "average precision": ap,
        "ap per class": ap_class,
        "ap 50": ap50, 
        "mp": mp, 
        "mr": mr, 
        "map50": map50, 
        "map": map
    }
    for metr in metric_res:
        metrics.append(metr)
        vals.append(metric_res[metr])
    results = pd.DataFrame({'metric':metrics, 'score':vals})
    results.to_csv("experiments/batched_val/{}_batch_{}_eval.csv".format(results_name, batch_size))

def secure_val():
    '''
    Takes as input the plaintext model, and encrypts the model using 
    CrypTen constructions. A secondary set of processes are established
    to compute a forward pass on each batch of the data. Batches of data
    are saved to .pth formats which can be loaded by CrypTen. 
    '''
    pass

def main():
    lbs_folder = "/mnt/data/coco128/coco128/labels/train2017"
    imgs_folder = "/mnt/data/coco128/coco128/images/train2017"
    
    # run plaintext validation for these folders
    plaintext_val(
        batch_size=32, 
        conf_thres=0.25,
        iou_thres=0.45, 
        max_det=1000, 
        classes=None, 
        agnostic_nms=None, 
        lbs_folder=lbs_folder, 
        imgs_folder=imgs_folder,
        model='yolov5s',
        device='cpu', 
        plain=True, 
        results_name='plaintext_val'
    )
    
    return 0    

def test_single():
    conf_thres=0.25        # confidence threshold parameter
    iou_thres=0.45           # iou threshold parameter
    max_det=1000           # max detection number
    classes=None            # indicates how to filter the classes
    agnostic_nms=None       # perofrms class agnostic nms (if True)
    
    lbs_folder = "/mnt/data/coco128/coco128/labels/train2017"
    imgs_folder = "/mnt/data/coco128/coco128/images/train2017"
    
    lbl_file = os.listdir(lbs_folder)[0]
    lbl_tensor = read_label_file("{}/{}".format(lbs_folder, lbl_file)) # load labels
    im_name = lbl_file.split(".")[0]
    
    for im in os.listdir(imgs_folder):  
        if im_name in im: 
            im_file = im # get correct image
            break
        
    im_tensor, og_size, new_size = read_img_file("{}/{}".format(imgs_folder, im_file)) # load image
    print("[DEBUG]: file name = {}".format("{}/{}".format(imgs_folder, im_file)))
    print("[DEBUG]: label name = {}".format("{}/{}".format(lbs_folder, lbl_file)))
    
    pln_model = torch.hub.load("ultralytics/yolov5", "yolov5s", force_reload=True, trust_repo=True)
    pln_names = pln_model.names
    
    start = time.time()
    pred = pln_model(im_tensor[None])
    end = time.time()
    
    pred = non_max_suppression( # obtain final bounding box predictions
        prediction=pred, 
        conf_thres=conf_thres, 
        iou_thres=iou_thres,
        classes=classes,
        agnostic=agnostic_nms, 
        max_det=max_det
    )
    
    metrics = []
    values = []
    
    stats = [] # initialize list for tracking metrics
    
    print("[DEBUG]: Pred output = {}".format(pred))

    iouv = torch.linspace(0.5, 0.95, 10, device="cpu")
    # pred[0][:,4] = pred[0][pred[0][:,4] > conf_thres] # get predictions above conf threshold (unecessary)
        
    correct = np.zeros((len(pred[0]), iouv.shape[0])).astype(bool)
    print("[INFO]: correct (og) = {}".format(correct))
    correct_class = lbl_tensor[:,0:1] == pred[0][:,5]
    
    print("[DEBUG] new_size={}, og_size={}".format(new_size, og_size))
    pred[0][:,:4] = scale_boxes(new_size, pred[0][:,:4], og_size).round() # rescale to proper image size
    #                               width     , height,   , width     , height
    print("[DEBUG] label tensor shape = {}".format(lbl_tensor.shape))
    print("[DEBUG] og size tensor = {}".format(og_size))
    lbl_tensor[:,1:] *= torch.tensor((og_size[1], og_size[0], og_size[1], og_size[0]))
    
    print("[DEBUG] type(lbl_tensor)={}".format(type(lbl_tensor[1:])))
    print("[DEBUG] type(pred)={}".format(type(pred[:3])))
    
    print("[DEBUG]: lbl_tensor = {}".format(lbl_tensor[:,1:]))
    print("[DEBUG]: pred = {}".format(pred[0][:,:4]))
    
    # if lbl_tensor[1:].shape != pred[0][:,:4].shape: 
    #     print("[!!ERROR!!]: mismatched tensor sizes. An inference error occurred")
    #     return
    
    iou_score = box_iou(lbl_tensor[:,1:], pred[0][:,:4])
    print("[INFO]: time = {} seconds".format(end-start))
    print("[INFO]: IOU = {}".format(iou_score))
    
    for i in range(len(iouv)):
        x = torch.where((iou_score > iouv[i]) & correct_class)
        print("[INFO] x = {}".format(x))
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x,1), iou_score[x[0],  x[1]][:, None]), 1).cpu().numpy()
            print("[DEBUG] (1) matches iter={} = {}".format(i, matches))
            if x[0].shape[0] > 1:
                matches = matches[matches[:,2].argsort()[::-1]]
                print("\t[DEBUG] (1a) matches = {}".format(matches))
                matches = matches[np.unique(matches[:,1], return_index=True)[1]]
                print("\t[DEBUG] (1b) matches = {}".format(matches))
                matches = matches[np.unique(matches[:,0], return_index=True)[1]]
                print("\t[DEBUG] (1c) matches = {}".format(matches))
            correct[matches[:,1].astype(int), i] = True
    correct = torch.tensor(correct, dtype=torch.bool, device=iouv.device) # convert numpy array into a tensor
    
    print("[INFO]: correct = {}".format(correct))
    stats.append((correct, pred[0][:,4], pred[0][:,5], lbl_tensor[:,0])) # (correct, conf, pcls, tcls)
    
    print("[DEBUG]: type(correct) = {}".format(type(correct)))
    print("[DEBUG]: type(pred[0][:,4]) = {}".format(pred[0][:,4]))
    print("[DEBUG]: type(pred[0][:,5]) = {}".format(pred[0][:,5]))
    print("[DEBUG]: type(lbl_tensor[:,0]) = {}".format(lbl_tensor[:,0]))
    
    # compute metrics
    stats = [torch.cat(x,0).cpu().numpy() for x in zip(*stats)] # convert these stats into tensors
    tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, plot=False, save_dir="experiments/batched_val", names=pln_names)
    ap50, ap = ap[:,0], ap.mean(1) # AP@0.5, AP@0.5:0.95
    mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
    
    # nc = 1 # number of image inferences is 1
    # nt = np.bincount(stats[3].astype(int), minlength=nc) 
    metric_res = {
        "true positive": tp, 
        "false positive": fp, 
        "precision": p, 
        "recall": r, 
        "f1 score": f1, 
        "average precision": ap, 
        "ap per class": ap_class, 
        "ap 50": ap50, 
        "mp": mp, 
        "mr": mr, 
        "map50": map50, 
        "map": map        
    }
    for metr in metric_res: 
        metrics.append(metr)
        values.append(metric_res[metr])
    results = pd.DataFrame({"metric":metrics, "score":values})
    results.to_csv("experiments/batched_val/single_eval.csv")
    
if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("spawn") # set thread start method for crypten 
    main()