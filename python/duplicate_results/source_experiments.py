import torch
import scipy.io as sio
import numpy as np
from sklearn.metrics import accuracy_score
import torch.optim as optim
from compact_cnn import compact_cnn
import os

torch.cuda.empty_cache()

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
def write_exp_results(file_name, results_var, test_subj):
    if not file_name in os.listdir():
        with open(file_name, 'w') as f:
            f.write(f'test_subj = {test_subj} : {results_var}\n')
    else:
        with open(file_name, 'a') as f:
            f.write(f'test_subj = {test_subj} : {results_var}\n')

def run(run_number, results_dict):
    '''
    run_number : used for output file formatting - source_experiment_results{run_number}
    results_dict : used to save accuracies for individual trials of an experiment
    '''
#    load data from the file

    filename = r'../data/dataset.mat' 
    
    tmp = sio.loadmat(filename)
    xdata=np.array(tmp['EEGsample'])
    label=np.array(tmp['substate'])
    subIdx=np.array(tmp['subindex'])

    label.astype(int)
    subIdx.astype(int)
    
    samplenum=label.shape[0]
    
#   there are 11 subjects in the dataset. Each sample is 3-seconds data from 30 channels with sampling rate of 128Hz. 
    channelnum=30
    subjnum=11
    samplelength=3
    sf=128
    
#   define the learning rate, batch size and epoches
    lr=1e-2 
    batch_size = 50
    n_epoch =6 
    
#   ydata contains the label of samples   
    ydata=np.zeros(samplenum,dtype=np.longlong)
    
    for i in range(samplenum):
        ydata[i]=label[i]

#   only channel 28 is used, which corresponds to the Oz channel
    selectedchan=[28]
    
#   update the xdata and channel number    
    xdata=xdata[:,selectedchan,:]
    channelnum=len(selectedchan)
    
#   the result stores accuracies of every subject     
    results=np.zeros(subjnum)
    
    
    
#   it performs leave-one-subject-out training and classfication 
#   for each iteration, the subject i is the testing subject while all the other subjects are the training subjects. 

#   TODO edited: for i in range(1,subjnum+1): --> only evaluates training on the 9th (test) subject at the moment    
 
    for i in range(9,10):
#       form the training data        
        trainindx=np.where(subIdx != i)[0] 
        xtrain=xdata[trainindx]   
        x_train = xtrain.reshape(xtrain.shape[0],1,channelnum, samplelength*sf)
        y_train=ydata[trainindx]
                
        
#       form the testing data         
        testindx=np.where(subIdx == i)[0]    
        xtest=xdata[testindx]
        x_test = xtest.reshape(xtest.shape[0], 1,channelnum, samplelength*sf)
        y_test=ydata[testindx]
    

        train = torch.utils.data.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
        train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)

#       load the CNN model to deal with 1D EEG signals
        my_net = compact_cnn().double().cuda()

   
        optimizer = optim.Adam(my_net.parameters(), lr=lr)    
        loss_class = torch.nn.NLLLoss().cuda()

        for p in my_net.parameters():
            p.requires_grad = True    
  
#        train the classifier 
        for epoch in range(n_epoch):   
            for j, data in enumerate(train_loader, 0):
                inputs, labels = data                
                
                input_data = inputs.cuda()
                class_label = labels.cuda()              

                my_net.zero_grad()               
                my_net.train()          
   
                class_output= my_net(input_data) 
                err_s_label = loss_class(class_output, class_label)
                err = err_s_label 
             
                err.backward()
                optimizer.step()

#       test the results
        my_net.train(False)
        with torch.no_grad():
            x_test =  torch.DoubleTensor(x_test).cuda()
            answer = my_net(x_test)
            probs=answer.cpu().numpy()
            preds       = probs.argmax(axis = -1)  
            acc=accuracy_score(y_test, preds)

# all experimental results are saved to a .txt file name "source_experiments_results.txt"
            #write_exp_results(f'source_experiments_results{run_number}.txt', results_var=acc, test_subj=i)
# save the model which generated this result to the proper file
            
            torch.save(my_net.state_dict(), f'model_subj_{i}.pth')
            print(acc)
            results[i-1]=acc
        results_dict[run_number] = float(acc)
            
    #print('mean accuracy:',np.mean(results))



if __name__ == '__main__':
    import math

    def set_patience_parameter(percent_difference, current_p_parameter):
        '''
        if the percent difference is less than 0.5 for 10
        rounds in a row, then terminate the sequence.

        However, if percent patience rises above 0.5 during
        one of those rounds, then reset the patience parameter
        '''
        if percent_difference < 0.1:
            current_p_parameter +=1 
        else: 
            current_p_parameter = 0
        return current_p_parameter

    def percent_dif(val1, val2):
        percent_dif = math.fabs(val2 - val1)/((val2 + val1)/2) * 100
        return percent_dif
    
    def write_3(file_handle, value_list, start_index=0):
        index = start_index
        count = 0
        while (count < 3) and (index < len(value_list)):
            if not count == 2: file_handle.write(f'{index:4}:{value_list[index]:25}|')
            else: file_handle.write(f'{index:4}:{value_list[index]:25}')
            index += 1
            count += 1
        file_handle.write('\n')

    def write_list_contents(list_obj, file_name):
        list_len = len(list_obj)
        iterations = list_len // 3 + 1
        index = 0
        with open(f'{file_name}', 'w') as f:
            f.write("="*95)
            f.write('\n')
            for i in range(iterations):
                write_3(f, list_obj, index)
                index += 3
            f.write("="*95)
            f.write('\n')

    def write_experiment_results(results_dict, file_name):
        with open (f'{file_name}', 'w') as f:
            for key in results_dict:
                f.write(f'{key}:{results_dict[key]}\n')
        
    # # run the source experiment 10 times
    results_dict = {}
    # # average accuracy between one trial and the next 
    # avg_over_trials = []
    # # int: total counter for the sum of all accuracies
    # total = 0
    # count = 0
    # # percent_dif tracker
    # percent_difference = 100
    # # initialize patience_parameter to 0, let run until
    # # the patience parameter reaches 10
    # patience_parameter = 0

    # while patience_parameter < 20:
    #     run_number = count
    #     # if not f'seed{run_number}' in os.listdir('test_subj9_iterative_results'):
    #     #     os.mkdir(f'test_subj9_iterative_results/seed{run_number}')
    #     # os.chdir(f'test_subj9_iterative_results/seed{run_number}')
    #     torch.manual_seed(run_number)
    #     run(run_number, results_dict)
    #     # os.chdir('../..')
        
    #     total += results_dict[run_number]
    #     count += 1
    #     avg_over_trials.append(total/count)
    #     if count > 1:
    #         percent_difference = percent_dif(avg_over_trials[run_number], avg_over_trials[run_number-1])
    #         patience_parameter = set_patience_parameter(percent_difference, patience_parameter)

    #         print(f'{count} -> {avg_over_trials[run_number]}, {avg_over_trials[run_number-1]}, {percent_difference} | {patience_parameter}')

    # write_experiment_results(results_dict, 'experiment_results6.txt')
    # write_list_contents(avg_over_trials, 'experiment_averages6.txt')
    run_number = 65
    torch.manual_seed(run_number)
    run(run_number, results_dict)
