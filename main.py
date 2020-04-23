rom cross_entropy import Cross_Entropy
from lag import LAG
from llsr import LLSR as myLLSR
from pixelhop2 import Pixelhop2
import numpy as np

from torch.utils.data.sampler import SubsetRandomSampler
from skimage.util import view_as_windows
import skimage.measure

import torch #for loading dataset
import torchvision 
import torchvision.transforms as transforms

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Normalizer

import timef

def load_train_data(fit_num=10000, trans_num=50000):
    print("start to load train data")
    #load the data using Pytorch lib
    transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])

    train = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    load_train = torch.utils.data.DataLoader(train, batch_size=1,
                                              shuffle=True, num_workers=2)
    
    #generate train indices for each class for subset extraction
    labels_list = [[] for _ in range(10)]
    count = 0
    images_50k = []
    labels_50k = []
    
    print("extracting 10 classes in 50k datas...")
    for images, labels in load_train:
        images_50k.append(images.numpy())
        labels_50k.append(labels.numpy())
        if (count % 10000 == 9999):
            print('{} / 50000'.format(count+1))
        category = np.asscalar(labels.numpy())
        labels_list[category].append(count)
        count += 1
        if count > 30000:
            break;
    print("extract finish")

    print("making fit datas...")
    train_indices = []
    [train_indices.extend(labels_list[i][0:int(fit_num/10)]) for i in range(10)]
    load_train_pixelhop2 = torch.utils.data.DataLoader(train, batch_size=1, sampler=SubsetRandomSampler(train_indices))
    images_fit = []
    for data in load_train_pixelhop2:
        images, labels = data
        images_fit.append(images.numpy())
    images_10karray = np.asarray(images_fit)
    X1 = images_10karray.reshape(fit_num, 3, 32, 32)
    X1 = np.moveaxis(X1, 1, -1)
    X1 = (X1+1)/2 #normalize
    print("successfully made fit dataset of size {}".format(X1.shape[0]))

    print("making transform datas...")
    
    if trans_num == 50000:
        X2 = np.asarray(images_50k).reshape(50000, 3, 32, 32)
        X2 = np.moveaxis(X2, 1, -1)
        X2 = (X2+1)/2 #normalize
        Y2 = np.asarray(labels_50k).ravel()
    else:
        train_indices = []
        [train_indices.extend(labels_list[i][0:int(trans_num/10)]) for i in range(10)]
        load_train_pixelhop2 = torch.utils.data.DataLoader(train, batch_size=1, sampler=SubsetRandomSampler(train_indices))
        images_trans = []
        labels_trans = []
        for data in load_train_pixelhop2:
            images, labels = data
            images_trans.append(images.numpy())
            labels_trans.append(labels.numpy())
        X2 = np.asarray(images_trans).reshape(trans_num, 3, 32, 32)
        X2 = np.moveaxis(X2, 1, -1)
        X2 = (X2+1)/2 #normalize
        Y2 = np.asarray(labels_trans).ravel()
    assert X2.shape[0] == Y2.shape[0], "The sizes of train data and label do not match!"
    print("successfully made fit dataset of size {}".format(X2.shape[0]))
    return X1, X2, Y2

def load_test_data(test_num=10000):
    print("start to load test data")
    #load the data using Pytorch lib
    transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    test = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    load_test = torch.utils.data.DataLoader(test, batch_size=1,
                                             shuffle=True, num_workers=2)
    
    images_test = []
    labels_test = []
    for images, labels in load_test:
        images_test.append(images.numpy())
        labels_test.append(labels.numpy())
    X_test = np.asarray(images_test)[0: test_num].reshape(test_num,3, 32, 32)
    X_test = np.moveaxis(X_test, 1, -1)
    X_test = Normalizer().fit(X_test).transformer.transform(X_test)
    Y_test = np.asarray(labels_test)[0: test_num].ravel()
    assert X_test.shape[0] == Y_test.shape[0], "The sizes of test data and label do not match!"
    print("successfully made test dataset of size {}".format(X_test.shape[0]))
    
    return X_test, Y_test

    def Shrink(X, shrinkArg):
        if int(X[0,:,0,0].shape[0]) <= 28:
            X = skimage.measure.block_reduce(X, (1,2,2,1), np.max)
        win = shrinkArg['win']
        stride = shrinkArg['stride']
        ch = X.shape[-1]
        X = view_as_windows(X, (1,win,win,ch), (1,stride,stride,ch))
        return X.reshape(X.shape[0], X.shape[1], X.shape[2], -1)

    # example callback function for how to concate features from different hops
def Concat(X, concatArg):
    return X


# feature selection
def feature_selection(layer1_o, layer2_o, layer3_o, Y):
    x_train1 = layer1_o.reshape(len(layer1_o), -1)
    x_train2 = layer2_o.reshape(len(layer2_o), -1)
    x_train3 = layer3_o.reshape(len(layer3_o), -1)
    input_list = [x_train1, x_train2, x_train3]
    res_list = []
    for i in range(3):
        print("Currently doing feature selection for hop {}".format(i+1))
        x_train = input_list[i]
        
        # Calculate loss for each features
        ce = Cross_Entropy(num_class=10, num_bin=5)
        feat_ce = np.zeros(x_train.shape[-1])
        start = time.time()
        for i in range(x_train.shape[-1]): #35280 features:
            if i % 1000 == 999:
                print("{}/{}".format(i+1, x_train.shape[1]))
            temp_vec = x_train1[:,i]
            feat_ce[i] = ce.KMeans_Cross_Entropy(x_train[:,i].reshape(-1,1), Y)
        end = time.time()
        print("time elapse: {}s".format(end-start))
        
        # Order the feature by its cross entropy loss
        feat_ce_tuplelist = []
        [feat_ce_tuplelist.append((feat_ce[i],i)) for i in range(x_train.shape[-1])]
        feat_ce_tuplelist.sort()
        
        # Select top 1000 features
        if len(feat_ce_tuplelist) > 1000:
            feat_ce_tuplelist_sel = feat_ce_tuplelist[0:1000]
            feat_num_sel = []
            [feat_num_sel.append(feat_ce_tuplelist_sel[i][1]) for i in range(1000)]
            x_train_sel = x_train[:,feat_num_sel]
            res_list.append(x_train_sel)
        else:
            res_list.append(x_train)
            
    return res_list[0], res_list[1], res_list[2]

def LAG_train_unit(x_1, x_2, x_3, Y):
    input_list =  [x_1, x_2, x_3]
    the_list = []
    lag_list = []
    for i in range(3):
        print("Currently doing LAG for hop {}".format(i+1))
        lag = LAG(encode='distance', num_clusters=[2,2,2,2,2,2,2,2,2,2], alpha=10, learner=myLLSR(onehot=False))
        lag.fit(input_list[i], Y)
        X_train_trans = lag.transform(input_list[i])
        the_list.append(X_train_trans)
        lag_list.append(lag)
    return the_list, lag_list

def LAG_test_unit(lag_list, x_1, x_2, x_3):
    input_list =  [x_1, x_2, x_3]
    the_list = []
    for i in range(3):
        print("Currently doing LAG for hop {}".format(i+1))
        lag = lag_list[i]
        X_train_trans = lag.transform(input_list[i])
        the_list.append(X_train_trans)
    return the_list

######################################### Preparation ##########################################
# set args
SaabArgs = [{'num_AC_kernels':-1, 'needBias':False, 'useDC':True, 'batch':None, 'cw': False}, 
            {'num_AC_kernels':-1, 'needBias':True, 'useDC':True, 'batch':None, 'cw': True},
            {'num_AC_kernels':-1, 'needBias':True, 'useDC':True, 'batch':None, 'cw': True}]
shrinkArgs = [{'func':Shrink, 'win':5, 'stride': 1}, 
              {'func':Shrink, 'win':5, 'stride': 1},
              {'func':Shrink, 'win':5, 'stride': 1}]
concatArg = {'func':Concat}

# Load train data
X1, X2, Y2 = load_train_data(10000, 10000) #(fit data size, transform data size), default(10000, 50000)

# new model
p2 = Pixelhop2(depth=3, TH1=0.001, TH2=0.001, SaabArgs=SaabArgs, shrinkArgs=shrinkArgs, concatArg=concatArg)



########################################### Module 1 ###########################################
print("Module 1 start")
start = time.time()

# Fit
print("Do fit on data of size {}".format(X1.shape))
p2.fit(X1)

# Get features (transform
print("Do transform on data of size {}".format(X2.shape))
output = p2.transform(X2)
end = time.time()
print("Module 1 time {} s".format(end-start))

########################################### Module 2 ###########################################
print("Module 2 start")
start = time.time()

# Max-pooling
print("do max-pooling")
layer1_o = skimage.measure.block_reduce((np.asarray(output[0])), (1,2,2,1), np.max) #add max pooling
layer2_o = skimage.measure.block_reduce((np.asarray(output[1])), (1,2,2,1), np.max) #add max pooling
layer3_o = np.asarray(output[2])
print("The output size of 3 hops are {}, {}, {}".format(layer1_o.shape, layer2_o.shape, layer3_o.shape))

# feature selection
print("do feature selection")
x_1, x_2, x_3 = feature_selection(layer1_o, layer2_o, layer3_o, Y2)

# LAG 
print("do LAG")
feat_hop_list, lag_list = LAG_train_unit(x_1, x_2, x_3, Y2)
end = time.time()
print("Module 2 time {} s".format(end-start))

########################################### Module 3 ###########################################
print("Module 3 start")
start = time.time()

# Concatenate
print("Do concatenate")
feat_final = np.concatenate((feat_hop_list[0], feat_hop_list[1]), axis=1)
feat_final = np.concatenate((feat_final, feat_hop_list[2]), axis=1)
print("The final features size {}".format(feat_final.shape))

# Train random forest classifier
print("Do classifier (Random Forest) training")
clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(feat_final, Y2)

# Test on train data
print("test on train data")
Y_pred = clf.predict(feat_final)
train_err = np.count_nonzero(Y2-Y_pred) / Y2.shape[0]
print("The traiing err is {}".format(train_err))
end = time.time()
print("Module 3 time {} s".format(end-start))

########################################### Testing ###########################################
# Load test data
X_test, Y_test = load_test_data(test_num=10000) #test data size, default 100000

# Get features (transform)
print("Do transform on data of size {}".format(X_test.shape))
output = p2.transform(X_test)

# Max-pooling
print("do max-pooling")
layer1_o = skimage.measure.block_reduce((np.asarray(output[0])), (1,2,2,1), np.max) #add max pooling
layer2_o = skimage.measure.block_reduce((np.asarray(output[1])), (1,2,2,1), np.max) #add max pooling
layer3_o = np.asarray(output[2])
print("The output size of 3 hops are {}, {}, {}".format(layer1_o.shape, layer2_o.shape, layer3_o.shape))

# feature selection
print("do feature selection")
x_1, x_2, x_3 = feature_selection(layer1_o, layer2_o, layer3_o, Y_test)

# LAG 
print("do LAG")
feat_hop_list = LAG_test_unit(lag_list, x_1, x_2, x_3)

# Concatenate
print("Do concatenate")
feat_final = np.concatenate((feat_hop_list[0], feat_hop_list[1]), axis=1)
feat_final = np.concatenate((feat_final, feat_hop_list[2]), axis=1)

# Test on train data
print("test on test data")
Y_pred = clf.predict(feat_final)
test_err = np.count_nonzero(Y_test-Y_pred) / Y_test.shape[0]
print("The testing err is {}".format(test_err))








