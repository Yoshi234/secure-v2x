import torch
import scipy.io as sio
import numpy as np
from sklearn.metrics import accuracy_score
import torch.optim as optim
# run from parent package
from python_models.compact_cnn import compact_cnn

torch.cuda.empty_cache()
torch.manual_seed(0)   

def print_torch_model_parameters(model):
    for name, value in model.state_dict().items():
        print(f"{name:20}:{value}")
        print(f"{'size':20}:{value.size()}\n")
        print(100*"=")

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
    print_torch_model_parameters(my_net)

    #Run the test code of the model
    my_net.train(False)
    
    # print the first 5 eeg_samples
    # for i in range(5):
    #     print(f"sample {i}")
    #     print(x_test[i])

    with torch.no_grad():
        temp_test = torch.DoubleTensor(x_test).cuda()
        answer = my_net(temp_test)
        probs = answer.cpu().numpy()
        preds = probs.argmax(axis = -1)
        acc = accuracy_score(y_test, preds)

        # results[test_subj] = acc
        results[0] = acc
        # print(preds)
        # print(y_test)
        print(acc)

        #save accuracy results to results.txt file

    # with open(f'{model_name}_results.txt', 'w') as f:
    #     f.write(f"{results}")
    #     f.write(f"\n")

if __name__ == "__main__":
    print("pretrained torch models are located at ../pretrained_torch_models/file.pth relative to this file")
    model_name = input("Please enter model name ('ELU' or 'relu'): ")
    subj_num = int(input("Please input the subject number: "))
    import os
    os.system('pwd')
    eval_model(model_name, subj_num) 