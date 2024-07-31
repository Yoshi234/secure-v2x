'''
Author: Yoshi234 7/17/2024

Detection process based on 'yolov5/detect.py' from Ultralytics.
See their script at https://github.com/ultralytics/yolov5/blob/master/detect.py 
Inference time implementation of Secure-RLR is based on CrypTen, and 
relies upon our custom implementation of several key functionalities. 

NOTE: had to move the following dirs from the yolov5 dir into the primary 
dir in order for dependencies to import correctly (yolov5)

+ data
+ models
+ utils
+ export.py

NOTE: disabling cuda can be done as follows:

```bash
conda env config vars set CONDA_VISIBLE_DEVICES=""
conda deactivate
conda activate crypten_env
```
'''

# save a set of images from one of the test videos and run it through
# the yolov5s object detector or just download a test image from the 
# yolov5 repository (preferred)

# utils
import os
import argparse
from utils.torch_utils import select_device
from utils.general import cv2, non_max_suppression, scale_boxes
from utils.augmentations import letterbox
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

SUPPORTED_IMG_TYPES = {"png", "jpg", ".jpeg"}
BUFFER_SIZE = 3000000 # calculated from shape of torch model output

def validate_onnx_yolo(mod_path='yolov5s.onnx'):
    model = onnx.load(mod_path)
    try: 
        onnx.checker.check_model(model)
        return True
    except onnx.checker.ValidationError:
        return False
    
def write_imgs(og_img, save_path, names:dict):
    '''
    For now, just write a single image
    '''
    annotator = Annotator(og_img, line_width=3, example=str(names))
    
def secure_run(
    weights='yolov5s.pt',               # path to the model weights
    pln_src="source/plaintext_source",  # (currently) supports a single image for inference
    cryp_src="source/crypten_source",   # (currently) supports a single image for inference
    img_file="img.png",                 # (currently) single image file for inference
    dataset="data/coco128.yaml",        # dataset .yaml file for yolo inference (labels)
    imgsz=(640,640),                    # input imgs are 640 x 640 px
    conf_thres=0.25,                    # confidence threshold
    iou_thres=0.45,                     # NMS IOU threshold for elminating bbs
    max_det=1000,                       # maximum detections per image
    device="1",                         # cuda or cpu device to run model on
    save_res=True,                      # bool indicate save formatted img or not
    secure=True,                        # bool indicating to run secure implementation of model
    hub=False,                          # bool indicating to use torch.hub.load model
    classes=None,                       # bool indicating how to filter class detections
    agnostic_nms=None,                  # bool (if true) performs class agnostic nms
    run_num=0,                          # label for the inference run
    model='yolov5s'
):
    """ 
    Runs the Yolov5s model in CrypTen. A separate (2-party process) 
    is called for evaluating the model in CrypTen itself
    
    pred output is of shape (1, num_anchor_boxes, num_classes+5)
    where each anchor box is of the shape (1, num_classes+5)
    each vector of shape num_classes+5 is given by 
    
    [x, y, width, height, objectness_score, class1_prob, class2_prob, ...,
    class num_classes_prob]
    
    x and y indicate the center of the candidate bounding box, while width and 
    height indicate the respective width and height of the bounding box. the 
    objectness score indicates the confidence of the bounding box, while
    the class probabilities indicate the classification score for each class. 
    I.e. the highest scoring class is the one which the object contained within 
    the bounding box is classified as.
    """
    # disables OpenMP threads -- needed by @mpc.run_multiprocess which uses fork
    # torch.set_num_threads(1) 
    device=select_device(device)
    
    # does this require nms on the output? I don't think so
    hub_model = torch.hub.load('ultralytics/yolov5', model, force_reload=True, trust_repo=True)
    hub_model = hub_model.to(device) # transfer to GPU
    hub_names = hub_model.names

    # load model with weights
    model = DetectMultiBackend(weights, device=device, dnn=False, data=dataset, fp16=False)
    stride, names, pt = model.stride, model.names, model.pt

    print("[DEBUG]: Auto = {}".format(pt))
    
    # load image file numpy array
    if img_file.split(".")[-1] in SUPPORTED_IMG_TYPES: 
        file_name = img_file.split(".")[0]
    else: 
        raise ValueError("Unsupported file type was passed")
    
    # read the original image
    im0 = cv2.imread("{}/{}".format(pln_src, img_file))
    
    print("[DEBUG]: image 0 size = {}".format(im0.shape))
    
    im = letterbox(im0, imgsz, stride=stride, auto=False, scaleFill=True)[0]
    im = im.transpose((2, 0, 1))[::-1]
    im = np.ascontiguousarray(im)
    
    print("[DEBUG]: image input size = {}".format(im.shape))

    # save image as a torch .pth file for use with CrypTen
    image_tensor = torch.from_numpy(im)/255
    torch.save(image_tensor, "{}/{}.pth".format(cryp_src, file_name))
    
    # send image to the correct device
    image_tensor = image_tensor.to(model.device)
    
    # warm up the GPU to improve running times
    model.warmup(imgsz=(1,3,*imgsz))
    
    # doesn't use the *visualize* feature of the model, just passes the image input
    if secure and (not hub): # NOTE: this option won't work. Do not use it 
        _ = encrypt_run(
            plaintext_model=model, 
            dat_path="{}/{}.pth".format(cryp_src, img_file.split(".")[0]), 
            imgsz=imgsz,
            run=run_num
        )
        
        # wait on the results to load then open them
        folder = "experiments/sec_outs"
        results_file = "run_{}.pkl".format(run_num)
        incomplete = True
        while incomplete:
            if results_file in os.listdir(folder): incomplete=False
            else: 
                print("[INFO-main]: Waiting on Exec.")
                time.sleep(30)
                
        with open("experiments/sec_outs/run_{}.pkl".format(run_num), 'rb') as f:
            res_pkl = pickle.load(f)
        pred, start, end = res_pkl # should be a list of this format
        
        with open("experiments/sec_outs/run_{}.pkl".format(run_num), 'rb') as f:
            res_pkl = pickle.load(f)
        pred, start, end = res_pkl # should be a list of this format
        
    elif secure and hub:
        # _ = encrypt_run(
        #     plaintext_model=hub_model,
        #     dat_path="{}/{}.pth".format(cryp_src, img_file.split(".")[0]),
        #     imgsz=imgsz,
        #     run=run_num
        # )
        
        # # wait on the results to load then open them
        # folder = "experiments/sec_outs"
        # results_file = "run_{}.pkl".format(run_num)
        # incomplete = True
        # while incomplete:
        #     if results_file in os.listdir(folder): incomplete=False
        #     else: 
        #         print("[INFO-main]: Waiting on Exec.")
        #         time.sleep(30)
                
        # with open("experiments/sec_outs/run_{}.pkl".format(run_num), 'rb') as f:
        #     res_pkl = pickle.load(f)
        # pred, start, end = res_pkl # should be a list of this format
        yolo_args = {
            "world_size":2,
            "img_size":imgsz,
            "model": hub_model,
            "data_path":"source/crypten_source/walkway.pth",
            "run_label":run_num,
            "batch_size":1,
            "device": device
        }
        multiproc_gpu(_run_sec_model, 'gpu_attempt', args=yolo_args)
        with open("experiments/gpu_sec_outs/run_{}.pkl".format(run_num), 'rb') as f:
            res_pkl = pickle.load(f)
        pred, start, end = res_pkl # should be a list of this format
        print("[INFO]: inference time = {} seconds for img size = {}".format(end - start, imgsz))
        
    elif (not secure) and hub:
        start = time.time()
        pred = hub_model(image_tensor[None])
        end = time.time()
        print("[DEBUG]: Size of pred output = {}".format(pred.shape))
        
        # DEBUG
        print(pred)

    else: # NOTE: this won't work for the secure version. just use the hub model
        start = time.time()
        pred = model(image_tensor[None])
        end = time.time()
        
        print("[DEBUG]: pred list items")
        print(pred[0])
        print("out shape = {}".format(pred[0].shape))
        
        # perform non-max suppression for multi backend model
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        for item in pred: print(item)
        
    print("[INFO]: torch hub model output\n{}".format(pred))
    # print("[INFO]: output shape = {}".format(pred.shape))
    # print("[DEBUG]: raw prediction output={} - time={}".format(len(pred), end-start))
    # print("\t[DEBUG]: output [0]:\n\t {}\n\t (shape)={}".format(pred[0], pred[0].shape))
    # print("\t[DEBUG]: output [1]:\n\t (len)={}".format(len(pred[1])))
    # print("\t\t[DEBUG]: output [1][0]:\n\t {}\n\t (shape)={}".format(pred[1][0], pred[1][0].shape))
    # print("\t\t[DEBUG]: output [1][1]:\n\t {}\n\t (shape)={}".format(pred[1][1], pred[1][1].shape))
    # print("\t\t[DEBUG]: output [1][2]:\n\t {}\n\t (shape)={}".format(pred[1][2], pred[1][2].shape))
    
    # TODO perform NMS on the output.
    
    # TODO add bounding box to the image and save it for viewing / analysis purposes
    
    return

def _run_sec_model(args:dict):
    '''
    Use the args parameter as a dictionary for holding key argument 
    variables for the yolo runs
    '''
    # import function to run the independent processes
    from twopc_yolo import run_2pc_yolo
    
    level = logging.INFO
    if "RANK" in os.environ and os.environ["RANK"] != "0":
        level = logging.CRITICAL
    logging.getLogger().setLevel(level)
    
    # pass all of the arguments
    run_2pc_yolo(args)
    
    return 

def multiproc_gpu(run_experiment, run_val='0', args:dict=None):
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
    
# @mpc.run_multiprocess(world_size=2) # Works for CPU
def encrypt_run(plaintext_model, dat_path, imgsz, run=0):
    '''
    Right now, we only load one image for inference, just as a test case
    
    returns the model.forward() output. non-maximum suppression of the 
    output bounding boxes should be performed locally
    '''
    ALICE=0
    BOB=1
    
    print("[INFO]: Crypten Initialization Beginning")
    crypten.init()
    dummy_input = torch.empty(1,3,*imgsz)
    
    # construct secure model
    sec_model = crypten.nn.from_pytorch(plaintext_model, dummy_input)
    sec_model.encrypt(src=ALICE)
    sec_model.eval()
    print("[INFO]: Model successfully encrypted = {}".format(sec_model.encrypted))
    
    data_enc = crypten.load_from_party(dat_path, src=BOB).reshape(1,3,*imgsz)
    
    assert data_enc.shape == torch.empty([1,3,*imgsz]).size()
    print("[DEBUG]: input data shape = {}".format(data_enc.shape))
    print("[INFO]: Sample data loaded from party (BOB)")
    
    print("[INFO]: Secure inference starting")
    start = time.time()
    pred_dec = sec_model(data_enc).get_plain_text()
    end = time.time()
    print("[INFO]: Secure inference finished")
    
    # return plaintext result of secure inference and timing results
    # return list size is too large, and so is getting hung. the 
    # size of this list should be expanded to allow for more 
    # data to be returned
    
    # pickle the output instead of returning is
    pkl_str = "experiments/sec_outs/run_{}.pkl".format(run)
    
    # only save if the file has not already been written by the other process
    if not "run_{}.pkl".format(run) in os.listdir("experiments/sec_outs"):
        with open(pkl_str, 'wb') as pkl_file:
            pickle.dump([pred_dec, start, end], pkl_file)
    
    return []
            
def sample_proc_output(imgsz:tuple=(640,640), folder="sec_outs", run_val=0, img_name='bus.jpg'):
    with open("experiments/{}/run_{}.pkl".format(folder, run_val), 'rb') as f:
        res_pkl = pickle.load(f)
    pred, start, end = res_pkl # should be a list of this format
    # shape of unprocessed output pred: 
    # [x center, y center, box width, box height, class 1 prob, ..., class n prob]
    pred = non_max_suppression(pred)
    # shape of NMS output pred: [[x1, y1, x2, y2, objectness score, class label]]
    # list of outputs corresponding to the batch size
    
    # unpack the outer singleton dimension
    pred = pred[0]
    
    # load model with names
    hub_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', force_reload=True, trust_repo=True)
    hub_names = hub_model.names
    
    # print("[INFO]: Type of the hub loaded model (which detection class) = {}".format(type(hub_model)))
    # print("[INFO]: Attributes of the loaded object = {}".format(dir(hub_model)))
    
    im0 = cv2.imread("{}/{}".format("source/plaintext_source", img_name))
    im0_copy = im0.copy()
    
    im = letterbox(im0, imgsz, stride=32, auto=False, scaleFill=True)[0]
    im = im.transpose((2, 0, 1))[::-1]
    im = np.ascontiguousarray(im)
    
    annotator = Annotator(im0_copy, line_width=3, example=str(hub_names))
    
    # scale the image to the correct size
    print("[DEBUG]: pred[:,:4] = {}".format(pred[:,:4]))
    print("[DEBUG]: im.shape[2:] = {}".format(im.shape[2:]))
    print("[DEBUG]: im0.shape = {}".format(im0.shape))
    print("[DEBUG]: im.shape = {}".format(im.shape))

    print("[DEBUG]: im.shape[1:] = {}".format(im.shape[1:]))
    pred[:,:4] = scale_boxes(im.shape[1:], pred[:,:4], im0.shape).round()

    # class filter for the thing
    select_classes = [1,2,3,5,7]
    for *xyxy, conf, cls in reversed(pred):
        c = int(cls)
        if not c in select_classes: continue # skip non vehicles
        # confidence = float(conf)
        # confidence_str = f"{confidence:.2f}"
        label = f'{hub_names[c]} {conf:.2f}'

        annotator.box_label(xyxy, label, color=colors(c, True))
    
    img_title = img_name.split(".")[0]
        
    cv2.imwrite("experiments/img_visuals/{}_boxes{}.jpg".format(img_title,run_val), im0_copy)
    
    print("[INFO]: Inference Duration = {} seconds".format(end-start))
      
# NOTE: The image size dimensions MUST be divisible
# by 32 or the execution will fail due to incorrect 
# sizes  
def main():
    mod='yolov5s'
    run = '320_320-walkway-cpu-{}'.format(mod)
    size = (192,320) # what happens with a square size for this image?
    img = 'walkway.png'
    secure_run(img_file=img, 
               device='cpu',
               model=mod, 
               imgsz=size,
               run_num=run,
               secure=True, 
               hub=True)
    sample_proc_output(imgsz=size, folder="gpu_sec_outs", run_val=run, img_name=img)
    
    # shrinking the image size from 640 x 640 to 480 x 480 
    # reduces inference time by a factor of 2 without affecting
    # the inference results - add experiments regarding resolution sizes
    # to the list of studies to conduct
    # it seems that reducing the resolution can actually 
    # improve confidence for certain image types
    # a 25% reduction in the size leads to a 2x reduction in processing
    # requirements. The image size resolution needs to be divisible by 32 though
    
    return
    
    # NOTE: Detections MUST be conducted in the same aspect
    # ratio or the prediction boxes will not scale correctly
    # maybe trying an image with a correctly scaled
    # frame - same aspect ratio will work better

    # increasing the size of the resolution of the 
    # image doesn't improve detection capability for a 
    # square aspect ratio projected onto the image

def export_detect_model():
    '''
    just ignore this bit - I don't feel like trying to figure 
    out if there is a non-loaded version of the base architecture
    class which can be used. Just hide the plaintext data. 
    
    check that this is the correct model
    '''
    from models.experimental import attempt_load
    from torch.onnx import OperatorExportTypes
    _OPSET_VERSION = 17
    
    kwargs = {
        "do_constant_folding": False,
        "export_params": True,
        "input_names": ["input"],
        "operator_export_type": OperatorExportTypes.ONNX,
        "output_names": ["output"],
        "opset_version": _OPSET_VERSION
    }
    
    model = attempt_load("yolov5s", map_location=torch.device("cpu"))
    torch_input = torch.rand((1,3,640,640))
    torch.onnx.export(model, torch_input, "yolov5s_detect.onnx", **kwargs)
      
    # CHECK THAT THE MODEL LOADS CORRECTLY
    state_dict = torch.load('yolov5s.pt')
    # model.load_state_dict(state_dict)
    # for key in state_dict: print(key)
    # print("[DEBUG]: State Dict = {}".format(state_dict))
    # print("[INFO]: Model loaded successfully")
    
if __name__ == "__main__":
    # if validate_onnx_yolo(): print("[INFO]: model is valid")
    # does the torch.hub.load model include nmx?
    
    mp.set_start_method("spawn")
    main()
    # export_detect_model()
    
    ## Resolution Reduction + Results
    ## 256 x 256 - 23.46 seconds
    ## 288 x 288 - 26.54 seconds
    ## 320 x 320 - 