### core dependency ###
import cv2
import caffe
import h5py

### system util ###
import os
import sys
import numpy as np
import argparse
from matplotlib import pyplot as plt

### util ###
from Data_Preperation import importFromTXT, exportH5PY
from MLP import *
sys.path.append('../')
from Util import Monitor


class ploter(object):
    def __init__(self, y):
        self.x = [i for i in range(len(y[0]))]
        self.y = y
        print(self.y)
        self.fig = plt.figure(1)
        plt.plot(self.x, y[0], label="0")
        plt.plot(self.x, y[1], label="1")
        plt.plot(self.x, y[2], label="2")
        plt.plot(self.x, y[3], label="3")
        plt.plot(self.x, y[4], label="4")
        plt.plot(self.x, y[5], label="5")
        plt.legend()
        plt.show()

#######################
### argument parser ###
#######################
ag = argparse.ArgumentParser()
ag.add_argument("-d", "--data-directory", required=False, help="path to the data file")    # format: -r 128,96
ag.add_argument("-t", "--processed-train", required=False, help="name of the train data file: default is 'train.h5'")  # format: -c 8  (row and colume are assume to be the same)
ag.add_argument("-v", "--processed-test", required=False, help="name of the test data file: default is 'test.h5'")  # format: -c 8  (row and colume are assume to be the same)
ag.add_argument("-tb", "--train-batch-size", required=False, help="parameters to define the batch size")                    # format: -tb 300  (300 samples per iteration)
ag.add_argument("-vb", "--test-batch-size", required=False, help="parameters to define the batch size")                 # format: -vb 100  (300 samples per iteration)
ag.add_argument("-e", "--epoch-num", required=False, help="parameters to specify the number of training epochs")         # format: -e 100 (defualt is 100)
args = vars(ag.parse_args())

# training parameters
train_batch_size = 300
test_batch_size = 100
epochs = 100
if args["train_batch_size"]:
    train_batch_size = int(args["train_batch_size"])
if args["test_batch_size"]:
    test_batch_size = int(args["test_batch_size"])
if args["epoch_num"]:
    epochs = int(args["epoch_num"])

# data path
if not os.path.exists('data'):
    os.makedirs('data')
if not os.path.exists('models'):
    os.makedirs('models')
cached_dataDir = os.path.join(os.getcwd(), 'data')
output_model_dir = os.path.join(os.getcwd(), 'models')


# finding train/test database file. If not given, default should be in /bin folder after running the C++ executable.
srcDataDir = os.path.join(os.getcwd(), "../../../bin")
train_data = "train.h5"
test_data = "test.h5"
if args["data_directory"]:
    srcDataDir = args["data_directory"]
if args["processed_train"]:
	train_data = args["processed_train"]
if args["processed_test"]:
	test_data = args["processed_test"]
train_data_filepath = os.path.join(srcDataDir, train_data)
test_data_filepath = os.path.join(srcDataDir, test_data)
print("using train data: ", train_data_filepath)
print("using test data: ", test_data_filepath)
train_file = h5py.File(train_data_filepath, 'r')
test_file = h5py.File(test_data_filepath, 'r')


train_data_len = len(train_file['data'])
feature_vector_len = len(train_file['data'][0])
test_data_len = len(test_file['data'])
print ("num of samples: ", train_data_len, "sample vector length: ", feature_vector_len)

if not os.path.exists(train_data_filepath) or not os.path.exists(test_data_filepath):
    print("[Error]: Expecting  processed train and test data file exists in the directory, but at least one is missing. Action aborted.")
    exit(1)


# prepare data source path for training
train_path_txt = os.path.join(cached_dataDir, 'train_path.txt')
test_path_txt = os.path.join(cached_dataDir, 'test_path.txt')
with open(train_path_txt, 'w') as f:
    print(train_data_filepath, file=f)
with open(test_path_txt, 'w') as f:
    print(test_data_filepath, file=f)



print("[Note]: Generate and compile autoencoder model...")
# model path
modelName = "MLP"
trainModel = os.path.join(output_model_dir, modelName+'_train.prototxt')
testModel = os.path.join(output_model_dir, modelName+'_test.prototxt')
inferenceModel = os.path.join(output_model_dir, modelName+'_deploy.prototxt')

### prepare model and sovler
#caffe.set_mode_cpu()
caffe.set_device(0)
caffe.set_mode_gpu()
with open(trainModel, 'w') as f:
    f.write(str(MLP(train_path_txt, train_batch_size, feature_vector_len, 'train')))

with open(testModel, 'w') as f:
    f.write(str(MLP(test_path_txt, test_batch_size, feature_vector_len, 'test')))

with open(inferenceModel, 'w') as f:
    f.write(str(MLP(test_path_txt, test_batch_size, feature_vector_len, 'inference')))

print("[Note]: Configuring the optimizer...")
solver_path = Solver(modelName, trainModel, testModel, train_data_len, train_batch_size, epochs, output_model_dir)
solver = caffe.get_solver(solver_path)

print("[Note]: Start training...")
### training loop
monitor = Monitor()
for e in range(epochs):
    print("starting new epoch...")
    solver.step(train_data_len//train_batch_size)

    print('epoch: ', e, 'testing...')
    loss = solver.net.blobs['loss'].data[()]                # train loss
    train_acc = solver.net.blobs['accuracy'].data[()]       # train accuracy
    #ploter(solver.net.blobs['data'].data)
    #variable = input('input something!: ')
    
    # test while in train and calculate accuracy
    correct = 0
    correct1, correct2, correct3 = 0, 0, 0
    for test_it in range(3):
        solver.test_nets[0].forward()
        correct += sum(solver.test_nets[0].blobs['prob'].data.argmax(1)
                       == solver.test_nets[0].blobs['label'].data.squeeze())

        #correct1 += sum(solver.test_nets[0].blobs['data'].data[:, 0:2].argmax(1)
        #                == solver.test_nets[0].blobs['label'].data.squeeze())
        #correct2 += sum(solver.test_nets[0].blobs['data'].data[:, 2:4].argmax(1)
        #                == solver.test_nets[0].blobs['label'].data.squeeze())
        #correct3 += sum(solver.test_nets[0].blobs['data'].data[:, 4:6].argmax(1)
        #                == solver.test_nets[0].blobs['label'].data.squeeze())
        #print(solver.test_nets[0].blobs['data'].data)
        #print(solver.test_nets[0].blobs['label'].data)

    """
    # write log to txt
    f = open("log1.txt", "a")
    num_of_samples = len(solver.test_nets[0].blobs['data'].data)
    for i in range(num_of_samples):
        softmaxs = [solver.test_nets[0].blobs['data'].data[i][:2], \
                    solver.test_nets[0].blobs['data'].data[i][2:4], \
                    #solver.test_nets[0].blobs['data'].data[i][4:6].argmax(0), \
                    solver.test_nets[0].blobs['prob'].data[i].argmax(0), \
                    solver.test_nets[0].blobs['label'].data[i]]
        f.write(" ".join(str(x) for x in softmaxs))
        f.write("\n")
    #accuracy1 = correct1/test_batch_size/3
    #accuracy2 = correct2/test_batch_size/3
    #accuracy3 = correct3/test_batch_size/3
    f.write("accuracy1: %f, accuracy2: %f\n" % (accuracy1, accuracy2))
    f.close()
    """

    #print(correct)
    test_acc = correct/test_batch_size/3
    monitor.update(loss, train_acc, test_acc)
    print("current loss: %f, current train accuracy: %f, current test accuracy: %f. " % (loss, train_acc, test_acc)) 

input('press "enter" to exit the program.')