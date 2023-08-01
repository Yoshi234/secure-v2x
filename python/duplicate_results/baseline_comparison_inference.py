# the test index for the subject 9 value is 284
import sys
sys.path.append("/home/jjl20011/snap/snapd-desktop-integration/current/Lab/V2V-Delphi-Applications/python/models")
sys.path.append("/home/jjl20011/snap/snapd-desktop-integration/current/Lab/V2V-Delphi-Applications/python/no_batch-norm")
import torch
import scipy.io as sio
import numpy as np
import torch.optim as optim
# path was appended above
from compact_cnn_no_batch_norm import compact_cnn_no_batch_norm

torch.cuda.empty_cache()
torch.manual_seed(4)

FILE = "/home/jjl20011/snap/snapd-desktop-integration/current/Lab/V2V-Delphi-Applications/python/no_batch-norm/subj9_seed0_no_batch_norm.pth"

def run_inference(subj_num=9, sample_idx=284):
    filename = r'../data/dataset.mat'

    # load data
    tmp = sio.loadmat(filename)
    xdata = np.array(tmp['EEGsample'])
    label = np.array(tmp['substate'])
    subIdx = np.array(tmp['subindex'])

    label.astype(int)
    subIdx.astype(int)

    samplenum = label.shape[0]
    
    # 3 second long clips
    samplelength = 3
    # each is sampled at 128 hz per second
    sf = 128

    # set up the labels
    ydata = np.zeros(samplenum, dtype=np.longlong)
    for i in range(samplenum):
        ydata[i] = label[i]

    selectedchan = [28]

    xdata = xdata[:,selectedchan,:]
    # we are only using 1 channel, so channelnum = 1
    channelnum = len(selectedchan)

    results = np.zeros(1)
    test_subj = subj_num


    # set up the test data
    testindx = np.where(subIdx == test_subj)[0]
    xtest = xdata[testindx]
    x_test = xtest.reshape(xtest.shape[0], 1, channelnum, samplelength*sf)
    # x_test is 314, 1, 1, 384 --> 314 samples, each of shape (1, 1, 384)
    y_test = ydata[testindx]

    x_test_val = x_test[284]
    x_test_val = np.expand_dims(x_test_val, axis=1)
    y_test_val = y_test[284]
    y_test_val = np.array(y_test_val)
    
    # x_test.reshape()

    my_net = compact_cnn_no_batch_norm().double().cuda()
    my_net.load_state_dict(torch.load(FILE))

    my_net.train(False)
    with torch.no_grad():
        temp_test = torch.DoubleTensor(x_test_val).cuda()
        answer = my_net(temp_test)
        probs = answer.cpu().numpy()
        preds = probs.argmax(axis=-1)

        print(f"Class was {y_test_val}, Inference result was {preds}")

if __name__ == "__main__":
    run_inference()