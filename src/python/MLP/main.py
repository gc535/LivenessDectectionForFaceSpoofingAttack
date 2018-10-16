### core dependency ###
import cv2
import caffe

### system util ###
import os
import sys
import numpy as np
import argparse

### util ###
from Data_Preperation import importFromTXT, exportH5PY
from MLP import *
sys.path.append('../')
from Util import Monitor


#######################
### argument parser ###
#######################
ag = argparse.ArgumentParser()
ag.add_argument("-d", "--data-directory", required=False, help="path to the data file")    # format: -r 128,96
ag.add_argument("-t", "--processed-train", required=False, help="name of the train data file: default is 'TrainFeature.txt'")  # format: -c 8  (row and colume are assume to be the same)
ag.add_argument("-v", "--processed-test", required=False, help="name of the test data file: default is 'TrainFeature.txt'")  # format: -c 8  (row and colume are assume to be the same)
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

srcDataDir = os.path.join(os.getcwd(), "../../../bin")
train_data_filename = "TrainFeature.txt"
train_label_filename = "TrainLabel.txt"
test_data_filename = "TestFeature.txt"
test_label_filename = "TestLabel.txt"
if args["data_directory"]:
    dataDir = args["data_directory"]
if args["processed_train"]:
	train_data_filename = args["processed_train"]
if args["processed_test"]:
	test_data_filename = args["processed_test"]
train_data_filepath = os.path.join(srcDataDir, train_data_filename)
train_label_filepath = os.path.join(srcDataDir, train_label_filename)
test_data_filepath = os.path.join(srcDataDir, test_data_filename)
test_label_filepath = os.path.join(srcDataDir, test_label_filename)
if not os.path.exists(train_data_filepath) or not os.path.exists(train_label_filepath) \
    or not os.path.exists(test_data_filepath) or not os.path.exists(test_label_filepath):
    print("[Error]: Expecting all processed (train/test) (data/label) file exists in the directory, but some are missing. Action aborted.")
    exit(1)


### load processed data and export to H5PY
print("[Note]: Loading processed training data from txt file to generate training set...")
train_data, train_data_len = importFromTXT(train_data_filepath)
train_label, train_label_len = importFromTXT(train_label_filepath)
assert(train_data_len == train_data_len)
trainSet = exportH5PY(train_data, train_label, train_data_filename[:-4], cached_dataDir)

print("[Note]: Loading processed testing data from txt file to generate testing set...")
test_data, test_data_len = importFromTXT(test_data_filepath)
test_label, test_label_len = importFromTXT(test_label_filepath)
assert(test_data_len == test_label_len)
testSet = exportH5PY(test_data, test_label, test_data_filename[:-4], cached_dataDir)

print("[Note]: Generate and compile autoencoder model...")
# model path
modelName = train_data_filename[:-4]+"_MLP"
trainModel = os.path.join(output_model_dir, modelName+'_train.prototxt')
testModel = os.path.join(output_model_dir, modelName+'_test.prototxt')

### prepare model and sovler
caffe.set_mode_cpu()
with open(trainModel, 'w') as f:
    f.write(str(MLP(trainSet, train_batch_size, 'train')))

with open(testModel, 'w') as f:
    f.write(str(MLP(testSet, test_batch_size, 'test')))

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
    loss = solver.net.blobs['loss'].data[()]
    correct = 0
    for test_it in range(3):
        solver.test_nets[0].forward()
        correct += sum(solver.test_nets[0].blobs['prob'].data.argmax(1)
                       == solver.test_nets[0].blobs['label'].data.squeeze())
    print(correct)
    accuracy = correct/test_batch_size/3
    monitor.update(loss, accuracy)
    print("current loss: %f, current accuracy: %f" % (loss, accuracy)) 

input('press "enter" to exit the program.')