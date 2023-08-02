use conda - set up the conda environment before beginning work
clone me to github
keep me private for now

## Directory Explanation
---
1. `\duplicate_results` contains files for replicatin the results from the paper: *A Compact and Interpretable Convolutional Neural Network for Cross-Subject Driver drowsiness Detection from Single-Channel EEG*

2. `\baseline_results` contains .txt files with the test accuracy for pytorch based models of `compact_cnn` using both `torch.nn.ReLU` and `torch.nn.ELU`

3. `\data` contains the dataset used in the paper's experiments (and this one) 

4. `\pretrained_torch_models` contains several `.pth` files for the `compact_cnn` models trained with both the `ReLU` and `ELU` functions as activation layers

5. Experiment Result Files: 
experiment_averages.txt - averages for training of different random seeds over the course of 100 trials (seeds 1 through 100 for full training and evaluation of the model on subject 9)

experiment_averages2.txt - averages for training of different random seeds until the percent difference between the average accuracy (between trials) is below 1 percent

experiment_averages3.txt - averages for training of different random seeds until the percent difference between the average accuracy (between trials) is below 0.5 percent

experiment_averages4.txt - averages for trianing of model on different random seeds until the percent difference between the average accuracy is below 0.5 percent for 10 trials in a row

experiment_averages5.txt - averages for training of model on different random seeds until the percent difference between average accuracy for sequential trials is below 0.5 percent for 20 trials in a row

experiment_averages6.txt - averages for training of model on different random seeds until the percent difference between average accuracy for sequential trials is below 0.1 percent for 20 trials in a row

6. Parameters Files: 
contains the parameters and their names loaded from the pth format (model state dict)

7. `\pretrained_numpy_models` 
contains the pretrained numpy weights and parameters of the models trained in python

8. `\no_batch-norm`
contains results of accuracy experiments when the model is trained and run without using batch normalization to center the weights of the model. This folder also contains the .pth file each model used for the experiment. The models were trained on the seed '1'
The file "experiment_results.txt" in this folder shows the results of the experiment, and the accuracy difference between the two runs
"experiment_results_2.txt" corresponds to the run where I used the seed '0' instead of seed '1' to see if a difference could be observed

9. `pytorch_batch_norm_tests` 
contains code for the 

