# SecureV2X

SecureV2X is a research framework for securely computing V2X 
applications. Namely, we design secure protocols for private object detection and red light violation detection
with YOLOv5n, and drowsiness detection via CompactCNN. To use our
framework, please follow these steps:

1. First, install `crypten` by running 
   
   ```
   pip install crypten
   ```

   CrypTen is not currently available through Conda and so pip must
   be used to install it.
   Our framework relies upon the use of secure primitives implemented by 
   CrypTen and so this package will be necessary to run *FastSec-YOLO*
   or *CryptoDrowsy*. Additionally, please make sure to install all of
   the packages listed in `requirements.txt`. 
2. Once `crypten` is installed, run the following in python
   
   ```python
   import crypten
   crypten.__file__
   ```

   This will output the location of `../crypten/__init__.py`. Navigate
   to the `nn` folder from there, and replace the default version of  `crypten/nn/module.py` with the `module.py` file provided
   at `case_studies/module.py` from our repository. Our version provides
   the necessary protocols to run SecureV2X. **NOTE: SecureV2X will not
   work correctly if this step is not followed correctly. Please take
   care to correctly locate the original `module.py` file and replace
   it with the updated code**

## CryptoDrowsy

## FastSec-YOLO

FastSec-YOLO can be run by doing the following:
First, navigate to `case_studies` and run `run_fastsec_yolo.py` with a command line 
argument of the form `source/image.jpg` where `image.jpg` is the image you want to 
perform secure object detection over. If you would like to perform inference over 
your own image, please upload that image to the source folder directly. Otherwise, 
you may test this functionality by running inference over the `bus.jpg` image 
which is held within the source folder. 

Once inference is performed, you may check the bounding box detection results 
in the `experiments/img_visuals` folder which will have been generated. 
The most likely issue to occur during this step is the improper initialization
of CUDA. If this happens, please check to make sure that your cuda settings 
are properly set. ⚠️ The code should try to handle potential issues automatically, 
but please be aware 

### Experiments

First, download the coco128 data from this [link](https://github.com/ultralytics/yolov5/releases/download/v1.0/coco128.zip). If the link does not work for some reason, please visit ultralytics/yolov5 on github and download
coco128.zip from release v1.0. 
Make a note of the absolute path to the image and label folders on your system,
or move the image and label data into the repository so that you can 
Our experiments for FastSec-YOLO can be reproduced as follows:

1. Navigate to `case_studies/traffic_rule_violation`
2. Run `python3 fastsec_yolo_val.py benchmark <labels_folder> <images_folder>` 
   where `<labels_folder>` contains the labels for the coco128 train image 
   set, and `<images_folder>` contains the actual images for the coco128 image
   dataset. 

The results of this experiment will be output to `case_studies/traffic_rule_violation/experiments/model_type_exps`
where each file in that folder is a `.csv` file containing the results of both plaintext
and secure inference over coco128 for each model (yolov5: n, s, m, l, x)

<!-- ## TODO: 8/5/24

1. Add the modified version of CrypTen to the repository with reproducible instructions
   to install the updated version of the code from source.
2. Clean up the crypten compactcnn implementation and convert to a script which can be run 
   more conveniently
3. Include instructions for accessing the data utilized for this work (make sure this is robust)
4. Update fully-automated RLR detection script, and move into the main v2x-delphi-2pc repo
   (the repo needs to be renamed as well to "v2x-2pc" or something)

For this code to work, we need to run all scripts from the case_studies package as a 
relative call now. This is because I have restructured everything as a package format. -->