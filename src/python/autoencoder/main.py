### core dependency ###
import cv2
import caffe

### system util ###
import os
import sys
import numpy as np
import argparse

### util ###
from Data_Preperation import importFromTXT, exportH5PY_featureonly
from autoencoder import *
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
train_filename = "TrainFeature.txt"
test_filename = "TestFeature.txt"
if args["data_directory"]:
    dataDir = args["data_directory"]
if args["processed_train"]:
	train_filename = args["processed_train"]
if args["processed_test"]:
	test_filename = args["processed_test"]
train_filepath = os.path.join(srcDataDir, train_filename)
test_filepath = os.path.join(srcDataDir, test_filename)

### load processed data and export to H5PY
print("[Note]: Loading processed training data from txt file to generate training set...")
train, train_vector_len = importFromTXT(train_filepath)
trainData = exportH5PY_featureonly(train, train_filename[:-4], cached_dataDir)
print("[Note]: Loading complete!")

print("[Note]: Loading processed testing data from txt file to generate testing set...")
test, test_vector_len = importFromTXT(test_filepath)
testData = exportH5PY_featureonly(test, test_filename[:-4], cached_dataDir)
print("[Note]: Loading complete!")
assert(train_vector_len == test_vector_len)

print("[Note]: Generate and compile autoencoder model...")
# model path
modelName = train_filename[:-4]+"Autoencoder"
trainModel = os.path.join(output_model_dir, modelName+'_train.prototxt')
testModel = os.path.join(output_model_dir, modelName+'_test.prototxt')

### prepare model and sovler
caffe.set_mode_cpu()
with open(trainModel, 'w') as f:
    f.write(str(createAutoencoder(trainData, train_vector_len, train_batch_size)))

with open(testModel, 'w') as f:
    f.write(str(createAutoencoder(testData, test_vector_len, test_batch_size)))

print("[Note]: Configuring the optimizer...")
solver_path = Solver(modelName, trainModel, testModel, output_model_dir)
solver = caffe.get_solver(solver_path)

print("[Note]: Model configuring complete!")
print("[Note]: Start training...")
### training loop
monitor = Monitor()
for e in range(epochs):
    print("starting new epoch...")
    solver.step(train_vector_len//train_batch_size)

    print('epoch: ', e, 'testing...')
    loss = solver.net.blobs['loss'].data[()]
    monitor.display_loss(loss)

input('press "enter" to exit the program.')