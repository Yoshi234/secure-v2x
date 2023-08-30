#architecture

#uses ELU function for the activation layers 
#(non-linear) instead of using the ReLU function

#we could try to perform NAS for this model with ReLU
#at a later date --> future work ;)

#also uses a "muted" ReLU function for activations
#in the 'final heatmap', whatever that means

#this file contains the model itself (implemented with torch)
import torch

#normalizelayer function definition for Batchlayer class
def normalizelayer(data):
    eps=1e-05
    a_mean = data - torch.mean(data, [0,2,3], True).expand(int(data.size(0)), int(data.size(1)), int(data.size(2)), int(data.size(3)))
    b = torch.div(a_mean, torch.sqrt(torch.mean((a_mean)**2, [0,2,3], True) + eps).expand(int(data.size(0)), int(data.size(1)), int(data.size(2)), int(data.size(3))))
    return b

#class definition for the batchLayer
class Batchlayer(torch.nn.Module):
    def __init__(self, dim):
        super(Batchlayer, self).__init__()
        self.gamma=torch.nn.Parameter(torch.Tensor(1,dim,1,1))
        self.beta=torch.nn.Parameter(torch.Tensor(1,dim,1,1))
        # random filling of the gamma / beta tensors with 
        # values in uniform distribution from -0.1 to 0.1
        self.gamma.data.uniform_(-0.1, 0.1)
        self.beta.data.uniform_(-0.1, 0.1)

    def forward(self, input):
        data=normalizelayer(input)
        #torch.Tensor.expand() returns view of self tensor with singleton dimensions
        #expanded to a larger size
        gammamatrix = self.gamma.expand(int(data.size(0)), int(data.size(1)), int(data.size(2)), int(data.size(3)))
        betamatrix = self.beta.expand(int(data.size(0)), int(data.size(1)), int(data.size(2)), int(data.size(3)))

        return data * gammamatrix + betamatrix


#torch.nn.Module: base class for all neural network modules
class compact_cnn_no_batch_norm(torch.nn.Module):
    """
    The codes implement the CNN model proposed in the paper "A Compact and Interpretable Convolutional Neural Network for-
    Cross-Subject Driver Drowsiness Detection from Single-Channel EEG ".
    
    The network is designed to classify 1D drowsy and alert EEG signals for the purposed of driver drowsiness recognition.
    
    Parameters:
        
    classes      : number of classes to classify, the default number is 2 corresponding to the 'alert' and 'drowsy' labels.
    Channels     : number of channels output by the first convolutional layer.
    kernelLength : length of convolutional kernel in first layer
    sampleLength : the length of the 1D EEG signal. The default value is 384, which is 3s signal with sampling rate of 128Hz.
    """
    def __init__(self, classes=2, channels=32, kernelLength=64, sampleLength=384):
        #initialize the parent class
        super(compact_cnn_no_batch_norm, self).__init__()
        self.kernelLength=kernelLength
        #kernel_size specifies the height and width of the convolution windows (1 by kernelLength here)
        self.conv = torch.nn.Conv2d(in_channels=1, out_channels=channels, kernel_size=(1,kernelLength))
        self.batch = Batchlayer(channels)
        # the shape of the average pooling layer is (1, 384-64+1=321) --> condenses it to 
        self.GAP = torch.nn.AvgPool2d((1, sampleLength- kernelLength + 1))
        # the linear layers map 32 input features to 2 output features (drowy and nondrowsy)
        self.fc = torch.nn.Linear(channels, classes)
        self.softmax = torch.nn.LogSoftmax(dim = 1)

    def forward(self, inputdata):
        #what is this nonsense with the self.conv() stuff? 
        # input.shape = [314, 1, 1, 384]
        intermediate = self.conv(inputdata)
        print(intermediate.shape)
        # intermediate.shape = [314, 32, 1, 321]
        # intermediate = self.batch(intermediate)
        # intermediate.shape = [314, 32, 1, 321]
        # compare the performance of the compactCNN with ReLU
        # versus with ELU function
        # intermediate = torch.nn.ELU()(intermediate)
        intermediate = torch.nn.ReLU()(intermediate)
        print(intermediate.shape)
        # intermediate.shape = [314, 32, 1, 321]
        intermediate = self.GAP(intermediate)
        print(intermediate.shape)
        # intermediate.shape = [314, 32, 1, 1]
        # this is just a reshape layer before the fully connected layer
        # squeezes the data into a matrix (314, 32) from (314, 32, 1, 1)
        intermediate.reshape(1, 32, 1, 1)
        intermediate = intermediate.view(intermediate.size()[0], -1)
        
        print(intermediate.shape)
        # intermediate.shape = [314, 32]
        intermediate = self.fc(intermediate)
        print(intermediate)
        # intermediate.shape = [314, 2]
        output = self.softmax(intermediate)

        return output
