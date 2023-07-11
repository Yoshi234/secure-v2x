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