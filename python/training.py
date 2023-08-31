# no need to set the GPU, none are in use at the moment
import torch
import scipy.io as sio
import numpy as np
from sklearn.metrics import accuracy_score
import torch.optim as optim
from python_models.compact_cnn import compact_cnn
from python_models.compact_cnn_approximation import compact_cnn_approximation

torch.cuda.empty_cache()
torch.manual_seed(1)

"""
 This file performs leave-one-subject cross-subject classification on the driver drowsiness dataset.
 THe data file contains 3 variables and they are EEGsample, substate and subindex.
 "EEGsample" contains 2022 EEG samples of size 20x384 from 11 subjects. 
 Each sample is a 3s EEG data with 128Hz from 30 EEG channels.

 The names and their corresponding index are shown below:
 Fp1, Fp2, F7, F3, Fz, F4, F8, FT7, FC3, FCZ, FC4, FT8, T3, C3, Cz, C4, T4, TP7, CP3, CPz, CP4, TP8, T5, P3, PZ, P4, T6, O1, Oz  O2
 0,    1,  2,  3,  4,  5,  6,  7,   8,   9,   10,   11, 12, 13, 14, 15, 16, 17,  18,  19,  20,  21,  22,  23,24, 25, 26, 27, 28, 29

 Only the channel Oz is used.

 "subindex" is an array of 2022x1. It contains the subject indexes from 1-11 corresponding to each EEG sample. 
 "substate" is an array of 2022x1. It contains the labels of the samples. 0 corresponds to the alert state and 1 correspond to the drowsy state.
 
  This file prints leave-one-out accuracies for each subject and the overall accuracy.
  The overall accuracy for one run is expected to be 0.7364. However, the results will be slightly different for different computers.
  
 If you have met any problems, you can contact Dr. Cui Jian at cuij0006@ntu.edu.sg
"""

def train_model(subIdx, xdata, channelnum, samplelength, sf, ydata, FILE,
                type="no_approx", n_epoch=6, batch_size=50, lr=1e-2, test_subj=8):
    """
    Initializes and trains the model - saving the weights and biases for each 
    layer to the file 'model.pth' in the working directory. These weights / biases
    can then be reloaded into scope for evaluation
    """
    trainindx = np.where(subIdx != test_subj)[0]
    xtrain = xdata[trainindx]
    x_train = xtrain.reshape(xtrain.shape[0], 1, channelnum, samplelength*sf)
    y_train = ydata[trainindx]


    train = torch.utils.data.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)

    #init my_net object
    my_net = None
    #load the CNN model to deal with 1D EEG signals
    if type == "no_approx":
        my_net = compact_cnn().double().cuda()
    elif type == "approx":
        my_net = compact_cnn_approximation().double().cuda()

    optimizer = optim.Adam(my_net.parameters(), lr=lr)
    #negative log likelihood loss
    loss_class = torch.nn.NLLLoss().cuda()

    for p in my_net.parameters():
        p.requires_grad = True
    
    # train the classifier
    for epoch in range(n_epoch):
        for j, data in enumerate(train_loader, 0):
            inputs, labels = data
            
            input_data = inputs.cuda()
            class_label = labels.cuda()

            my_net.zero_grad()
            my_net.train()

            class_output = my_net(input_data)
            err_s_label = loss_class(class_output, class_label)
            err = err_s_label

            err.backward()
            optimizer.step()

    #save the model as a torch pt (model) file
    #then reload the model in the main 'run_model' function
    # torch.save(my_net.state_dict(), FILE)

#train only on the 9th subject, and save the model weigths to the 
def run_model(model_name, type):
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
    subjnum = 11
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
    selectedchan = [28]

    #update xdata and channel number
    xdata = xdata[:,selectedchan,:]
    channelnum = len(selectedchan)

    #the result stores accuracies of every subject
    #size 11 array
    results = np.zeros(subjnum)

    #perform leave-one-subject-out training and classification
    
    #our training might involve performing a more even training
    #test split of the data. we would need more data to improve
    #the accuracy and efficiency of the model, but our work
    #is focused solely on comparing the effectiveness of the model
    #when we include Delphi and when we do not run the model using
    #delphi.

    #our implementation trains on subjects 1-8 and 10-11
    #tests the model on subject 9 (highest accuracy of experimental model)
    test_subj=9
    FILE = f'{model_name}.pth'
    train_model(subIdx, xdata, channelnum, samplelength, sf, ydata, FILE=FILE, test_subj=test_subj, type=type)

    #form the testing data
    testindx = np.where(subIdx == test_subj)[0]
    xtest = xdata[testindx]
    x_test = xtest.reshape(xtest.shape[0], 1, channelnum, samplelength*sf)
    y_test = ydata[testindx]

    #reload the model from saved file
    my_net = compact_cnn().double().cuda()
    #PATH should be set to the path to the state dict of the model
    
    my_net.load_state_dict(torch.load(FILE))
    
    #Run the test code of the model
    my_net.train(False)
    with torch.no_grad():
        # before, this line was redefining x_test every time. 
        # it starts out as non-cuda, and we make the tensor cuda() by calling
        # the cuda option on it, but it was already cuda to begin with
        x_test = torch.DoubleTensor(x_test).cuda()
        answer = my_net(x_test)
        probs = answer.cpu().numpy()
        preds = probs.argmax(axis = -1)
        acc = accuracy_score(y_test, preds)

        # results[test_subj] = acc

        #save accuracy results to results.txt file
        # with open(f'{model_name}_results.txt', 'w') as f:
        #     f.write(f"{acc}")
        
if __name__ == "__main__":
    model_name = input("Please enter the name of the model you would like to train and evaluate: ")
    model_type = input("Please enter the model type `approx` or `no_approx`")
    run_model(model_name = model_name, type=model_type)
    #I want to train / run two versions of the model - the model with ELU function 
    #and the model with the ReLU function implemented in its place



