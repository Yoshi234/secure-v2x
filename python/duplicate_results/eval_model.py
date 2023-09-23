from os import path
import torch
import scipy.io as sio
import numpy as np
from sklearn.metrics import accuracy_score
import torch.optim as optim
# run from parent package
from python_models.compact_cnn import compact_cnn
from python_models.compact_cnn_approximation import compact_cnn_approximation
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.utils import shuffle
torch.cuda.empty_cache()
torch.manual_seed(0)   

def save_data(batch_size, acc, data):
    ''' 
    Saves the current best position error and the particle number to a 
    pandas dataframe object

    Arguments:
    - err --- best error rate data to save
    - particle_number --- number of particles for given iteration
    - data --- list to save data to

    Returns: 
    data --- updated data
    '''
    data.append([batch_size, acc])
    return data

def graph_data(data, experiment_num):
    '''
    Graphs the best position error against particle number

    Arguments:
    - data --- `list` of 2-object-lists
    object from which to initialize the dataframe

    Returns:
    None
    '''
    df = pd.DataFrame(data, columns=["batch_size", "accuracy"])
    fig, ax = plt.subplots(figsize=(10,6))
    ax.scatter(x = df["batch_size"], y = df["accuracy"])
    plt.xlabel("Batch Size")
    plt.ylabel("Accuracy (proportion of correct predictions)")
    plt.savefig(f"batch_size_inference_experiments/multi_batch_size_test_{experiment_num}.png")

def load_model_data(subj_num):
    '''
    Loads the data used for inference

    Arguments:
    - subj_num --- indicates which subject to load test data from

    Returns:
    - `(x_test, y_test)` --- tuple containing the test
    datasets needed for inference / evaluation
    '''
    filename = r'data/dataset.mat'
    
    #function for loading into scope matlab files
    #returns a matlab:dict type with matrices as values (in key value pairs)
    tmp = sio.loadmat(filename)
    #strings EEGsample, substate, and subindex are 
    xdata = np.array(tmp['EEGsample'])
    label = np.array(tmp['substate'])
    subIdx = np.array(tmp['subindex'])

    label.astype(int)
    subIdx.astype(int)

    samplenum = label.shape[0]

    #11 subjects in the dataset, each sample being 3-seconds of 
    #data from 30 channels with a sampling rate of 128Hz
    channelnum = 30
    samplelength = 3
    sf = 128

    #labels of samples - a matrix of (2022,) shape, with 
    #type of np.longlong (64-bit signed integer type)
    #generates a matrix initialized to all zeros
    ydata = np.zeros(samplenum, dtype=np.longlong)

    for i in range(samplenum):
        ydata[i] = label[i]

    #single channel analysis is used for this paper, although
    #there is data corresponding to multiple channels
    #this is the 'oz' channel of the EEG data
    selectedchan = [28]

    #update xdata and channel number
    xdata = xdata[:,selectedchan,:]
    channelnum = len(selectedchan)

    #the result stores accuracies of every subject
    #size 11 array
    test_subj = subj_num
    FILE = f'{model_name}.pth'

    #form the testing data
    testindx = np.where(subIdx == test_subj)[0]
    xtest = xdata[testindx]
    x_test = xtest.reshape(xtest.shape[0], 1, channelnum, samplelength*sf)
    y_test = ydata[testindx]

    shuffle(x_test, y_test)
    return (x_test, y_test)

def print_torch_model_parameters(model):
    for name, value in model.state_dict().items():
        print(f"{name:20}:{value}")
        print(f"{'size':20}:{value.size()}\n")
        print(100*"=")

def sequential_inference_evaluation(model, subj_num=9):
    '''
    Performs inference on test data for a given `subj_num` 
    one-by-one `batch_size=1`
    
    Arguments:
    - model --- a pytorch model generated from `compact_cnn().double().cuda()
    - subj_num --- the number corresponding to the subject to use as the test
    dataset
    
    Returns: 
    - None
    '''
    filename = r'data/dataset.mat'
    
    #function for loading into scope matlab files
    #returns a matlab:dict type with matrices as values (in key value pairs)
    tmp = sio.loadmat(filename)
    #strings EEGsample, substate, and subindex are 
    xdata = np.array(tmp['EEGsample'])
    label = np.array(tmp['substate'])
    subIdx = np.array(tmp['subindex'])

    label.astype(int)
    subIdx.astype(int)

    samplenum = label.shape[0]

    #11 subjects in the dataset, each sample being 3-seconds of 
    #data from 30 channels with a sampling rate of 128Hz
    channelnum = 30
    samplelength = 3
    sf = 128

    #define learnign rate, batch size and epoches
    
    #learning rate = lr
    lr = 1e-2
    batch_size = 50
    #based on experimental results, the proposed model was
    #optimized after 6 training epochs
    n_epoch = 6 

    #labels of samples - a matrix of (2022,) shape, with 
    #type of np.longlong (64-bit signed integer type)
    #generates a matrix initialized to all zeros
    ydata = np.zeros(samplenum, dtype=np.longlong)

    for i in range(samplenum):
        ydata[i] = label[i]

    #single channel analysis is used for this paper, although
    #there is data corresponding to multiple channels
    #this is the 'oz' channel of the EEG data
    selectedchan = [28]

    #update xdata and channel number
    xdata = xdata[:,selectedchan,:]
    channelnum = len(selectedchan)

    #the result stores accuracies of every subject
    #size 11 array
    results = np.zeros(1)
    test_subj = subj_num
    FILE = f'{model_name}.pth'

    #form the testing data
    testindx = np.where(subIdx == test_subj)[0]
    xtest = xdata[testindx]
    x_test = xtest.reshape(xtest.shape[0], 1, channelnum, samplelength*sf)
    y_test = ydata[testindx]

    #reload the model from saved file
    my_net = compact_cnn().double().cuda()

    my_net.load_state_dict(torch.load(FILE))
    # print_torch_model_parameters(my_net)

    #Run the test code of the model
    my_net.train(False)
    
    # print the first 5 eeg_samples
    # for i in range(5):
    #     print(f"sample {i}")
    #     print(x_test[i])
    correct = []
    with torch.no_grad():
        for i in range(len(y_test)):
            temp_test = torch.DoubleTensor(x_test[i]).reshape(1, 1, 1, 384).cuda()
            answer = my_net(temp_test)
            print(answer)
            probs = answer.cpu().numpy()
            preds = probs.argmax(axis = -1)
            correct += [1] if preds == y_test[i] else [0]
    print(100*(sum(correct) / len(y_test)))

def multi_shuffle_experiments(model_name, subj_num=9):
    np.random.seed(0)
    for i in range(2, 6):
        multiple_batch_size_experiments(model_name, subj_num, experiment_num=i)

def multiple_batch_size_experiments(model_name, subj_num=9, experiment_num=0):
    batch_sizes = {1, 5, 10, 30, 50, 100, 150, 200, 250, 270, 314}
    data = []
    (x_test, y_test) = load_model_data(subj_num)
    for batch_size in batch_sizes:
        acc = varied_batch_size_inference(model_name, batch_size, x_test, y_test)
        data = save_data(batch_size, acc, data)
    graph_data(data, experiment_num)

def varied_batch_size_inference(model_name, batch_size:int, x_test, y_test, model_type="compact_cnn", subj_num=9):
    '''
    Performs inference on the model using varying batch sizes
    to determine whether batch size is the issue causing accuracy to drop
    so drastically for `sequential_inference_evaluation`
    
    Arguments:
    - model_name --- the pytorch model / path to use to load the model weights
    - batch_size --- the number of eeg_samples to include in each inference
    - model_type --- either `compact_cnn` or `compact_cnn_approximation`
        - the normal version uses `gc` to compute ReLU activations whereas
        the approximation uses quadratic approximations to compute the ReLU function
    - subj_num --- number of the subject upon which to perform inference

    Returns:
    - accuracy --- the accuracy of the model for a given batch size during inference
    '''
    FILE = f"{model_name}.pth"

    my_net = None
    if model_type =="compact_cnn": 
        my_net = compact_cnn().double().cuda()
    elif model_type == "compact_cnn_approximation":
        my_net = compact_cnn_approximation().double().cuda()

    my_net.load_state_dict(torch.load(FILE))

    # get the batches as a list of 'mostly' equally-sized batches
    batches = []
    batch_classes = []
    extra_samples = len(x_test) % batch_size
    for i in range(0, len(x_test)-extra_samples, batch_size):
        # if i is at the end, then just add all data
        if i == len(x_test) - batch_size - 1: 
            batches.append(x_test[i:])
            batch_classes.append(y_test[i:])
        else: 
            batches.append(x_test[i:i+batch_size])
            batch_classes.append(y_test[i:i+batch_size])
    
    # check that the lists are the same length
    assert len(batches) == len(batch_classes)

    correct = []
    # run the batches iteratively for inference
    for j in range(len(batches)):
        with torch.no_grad():
            temp_test = torch.DoubleTensor(batches[j]).cuda()
            answer = my_net(temp_test)
            probs = answer.cpu().numpy()
            preds = probs.argmax(axis = -1)
            # print(batch_classes[j])
            # print(preds)
            # print(40*"-")
            if batch_size > 1:
                for k in range(len(preds)):
                    correct += [1] if preds[k] == batch_classes[j][k] else [0]
            else:
                correct += [1] if preds[0] == batch_classes[j][0] else [0]

    acc = sum(correct) / len(y_test)
    # print(acc)
    # return the accuracy of the model for given batch size
    return acc
    
def eval_model(model_name, subj_num):
    filename = r'data/dataset.mat'
    
    #function for loading into scope matlab files
    #returns a matlab:dict type with matrices as values (in key value pairs)
    tmp = sio.loadmat(filename)
    #strings EEGsample, substate, and subindex are 
    xdata = np.array(tmp['EEGsample'])
    label = np.array(tmp['substate'])
    subIdx = np.array(tmp['subindex'])

    label.astype(int)
    subIdx.astype(int)

    samplenum = label.shape[0]

    #11 subjects in the dataset, each sample being 3-seconds of 
    #data from 30 channels with a sampling rate of 128Hz
    channelnum = 30
    samplelength = 3
    sf = 128

    #define learnign rate, batch size and epoches
    
    #learning rate = lr
    lr = 1e-2
    batch_size = 50
    #based on experimental results, the proposed model was
    #optimized after 6 training epochs
    n_epoch = 6 

    #labels of samples - a matrix of (2022,) shape, with 
    #type of np.longlong (64-bit signed integer type)
    #generates a matrix initialized to all zeros
    ydata = np.zeros(samplenum, dtype=np.longlong)

    for i in range(samplenum):
        ydata[i] = label[i]

    #single channel analysis is used for this paper, although
    #there is data corresponding to multiple channels
    #this is the 'oz' channel of the EEG data
    selectedchan = [28]

    #update xdata and channel number
    xdata = xdata[:,selectedchan,:]
    channelnum = len(selectedchan)

    #the result stores accuracies of every subject
    #size 11 array
    results = np.zeros(1)
    test_subj = subj_num
    FILE = f'{model_name}.pth'

    #form the testing data
    testindx = np.where(subIdx == test_subj)[0]
    xtest = xdata[testindx]
    x_test = xtest.reshape(xtest.shape[0], 1, channelnum, samplelength*sf)
    y_test = ydata[testindx]

    #reload the model from saved file
    my_net = compact_cnn().double().cuda()
    #PATH should be set to the path to the state dict of the model

    my_net.load_state_dict(torch.load(FILE))
    # print_torch_model_parameters(my_net)

    #Run the test code of the model
    my_net.train(False)
    
    # print the first 5 eeg_samples
    # for i in range(5):
    #     print(f"sample {i}")
    #     print(x_test[i])
    preds=None
    with torch.no_grad():
        temp_test = torch.DoubleTensor(x_test).cuda()
        answer = my_net(temp_test)
        print(answer)
        probs = answer.cpu().numpy()
        preds = probs.argmax(axis = -1)
        acc = accuracy_score(y_test, preds)

        # results[test_subj] = acc
        results[0] = acc
        # print(preds)
        # print(y_test)
        print(acc)

    correct = []
    # iterate through the predictions and save the results
    for i in range(len(preds)):
        correct += [1] if preds[i] == y_test[i] else [0]
    np.save("plaintext.npy", np.array(correct))
    print(100*(sum(correct)/len(y_test)))

if __name__ == "__main__":
    print("pretrained torch models are located at ../pretrained_torch_models/file.pth relative to this file")
    model_name = input("Please enter model name ('ELU' or 'relu'): ")
    subj_num = int(input("Please input the subject number: "))

    # import os
    # os.system('pwd')

    multi_shuffle_experiments(model_name, subj_num)

    # eval_model(model_name, subj_num) 
    # sequential_inference_evaluation(model_name, subj_num)


