#test-train-inference
#train, test and save the model in .pt foramt
#load the saved model and run some inferences
#use cuda for gpu acceleration

#Experimentl Setup:
#1. data from one subject - used for testing
#2. data from all other subjects used for training
#process is iterated until every subject has served at least once as the
#test subject

#perhaps a contribution of the work can be the construction of 
#the ELU activation protocol (garbled circuit) for use with Delphi
#for comparison with the original experiment, use batch_size of 50,
#the batch size is the number of training samples that are fed to the
#neural network at once

#the training epoch number is the number of times that the entire
#training dataset is passed through the network

#using mini-batches will reduce the risk of getting stuck at a local
#minimum since different batches will be considered at each iteration,
#granting robust convergence