### core dependency ###
import cv2
import lmdb
import h5py
import caffe

### system util ###
import os
import sys
import numpy as np

### util ###
sys.path.append('../')
from Util import * 
#from LBP_Feature_Extraction import *


#############################
### prepare training data ###
#############################

### save organized train/test into txt file for future use ###
def prepareData(action, resize_row, resize_col, cellSize, modelName, _oneHot=False):

    cur_dir = os.getcwd()
    par_dir = os.path.join(cur_dir, os.pardir)

    data_x = []  # feature vector to be returned
    data_y = []  # label vector to be returned

    if  os.path.exists(os.path.join(cur_dir, modelName+'_'+action+'_data.txt')) \
        and os.path.exists(os.path.join(cur_dir, modelName+'_'+action+'_label.txt')):
        # if data are already prepared, then just load it
        sys.stdout.write(action+"ing data already exists in this directory, load it from the file...")
        sys.stdout.flush()
        print(os.path.join(cur_dir, modelName+'_'+action+'_data.txt'))
        data_x = np.loadtxt(os.path.join(cur_dir, modelName+'_'+action+'_data.txt'), dtype='f')
        data_y = np.loadtxt(os.path.join(cur_dir, modelName+'_'+action+'_label.txt'), dtype=int).reshape(-1, 1)
    else:
        (progress_tracker, FILE_PER_PERCENT) = progress_bar_util(action)  # set up the progress bar
        pre_precent = 0
        # compute each feature vector 
        for label in range(4):    # directory idx used as tag
            gallery_path = os.path.join(par_dir, action, str(label))
            #gallery_path = par_dir+'/train/'+str(label)
            imgFiles = listimages(gallery_path)
            for file in imgFiles:
                # update prcenetage
                progress_tracker += 1
                if progress_tracker // FILE_PER_PERCENT > pre_precent:
                    pre_precent = progress_tracker // FILE_PER_PERCENT
                    sys.stdout.write("=")
                    sys.stdout.flush()
                    #progress_tracker = 0

                img = cv2.imread(os.path.join(gallery_path, file))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                data_x.append(computeLFBFeatureVector_rotation_uniform(img, (cellSize, cellSize), size=(resize_row, resize_col), doCrob=False))
                if _oneHot:
                    onehot_encode = [0, 0, 0, 0]
                    onehot_encode[label] = 1
                    data_y.append(onehot_encode)
                else:
                    data_y.append([label])

        data_x = np.array(data_x, dtype='f')
        data_y = np.array(data_y, dtype=int)
        data_x, data_y = ShuffleWithLabel(data_x, data_y)
        np.savetxt(os.path.join(cur_dir, modelName+'_'+action+'_data.txt'), data_x, fmt='%f')
        np.savetxt(os.path.join(cur_dir, modelName+'_'+action+'_label.txt'), data_y, fmt='%d')

    sys.stdout.write("\ncomplete!\n")
    sys.stdout.write(action+" feature vector shape: ")
    sys.stdout.flush()
    print(data_x.shape)
    sys.stdout.write(action+" label vector shape: ")
    sys.stdout.flush()
    print(data_y.shape)

    return data_x, data_y


### concatenate and shuffle ndarray ###
def ShuffleWithLabel(data_x, data_y):
    merged = np.concatenate((data_x, data_y), axis=1)
    np.random.shuffle(merged)
    data_x, data_y = np.hsplit(merged, [-1])
    return data_x, data_y


### export data to hdf5 file ###
def exportH5PY(data_x, data_y, fileName, outputDir=None):
    assert len(data_x) != 0 and len(data_x)==len(data_y)

    output_dir = os.getcwd()
    if outputDir:
        output_dir = outputDir
        
    h5pyFile = os.path.join(output_dir, fileName+'.h5')
    txtFile = os.path.join(output_dir, fileName+'_path.txt')

    if os.path.exists(h5pyFile):
        print('[Note]: Target output file: "'+h5pyFile+'" found in the current directory. Continuing... ')
    else:
        with h5py.File(h5pyFile, 'w') as f:
            f['data'] = data_x
            f['label'] = data_y

        with open(txtFile, 'w') as f:
            print(h5pyFile, file=f)

    return txtFile


### export data to hdf5 file ###
def exportH5PY_featureonly(data_x, fileName, outputDir=None):
    assert len(data_x) != 0
    output_dir = os.getcwd()
    if outputDir:
        output_dir = outputDir

    h5pyFile = os.path.join(output_dir, fileName+'.h5')
    txtFile = os.path.join(output_dir, fileName+'_path.txt')

    if os.path.exists(h5pyFile):
        print('[Note]: Target output file: "'+h5pyFile+'" found in the current directory. Continuing... ')
    else:
        with h5py.File(h5pyFile, 'w') as f:
            f['data'] = data_x

        with open(txtFile, 'w') as f:
            print(h5pyFile, file=f)

    return txtFile


### import data from txt file ###
def importFromTXT(filepath):

    if  os.path.exists(filepath):
        data = np.loadtxt(filepath, dtype='f')
        assert(len(data)!=0 and  len(data[0])!=0)
        print("[Note]: Number of samples: ", len(data), ", feature vector length: ", len(data[0]))
        return data, len(data[0])

    else:
        print("[ERROR]: No datafile found in ./bin. Action aborted.")
        exit()

### note: this mehtod currently is not fully functional ###
###########################################################
def exportLMDB(data_x, data_y, fileName):

    assert len(data_x) != 0 and len(data_x)==len(data_y) 

    if os.path.exists(os.path.join(os.getcwd(), fileName)):
        print('[Note]: Target output file: '+' found in the current directory. Continuing... ')
    else:
        n_samples = len(data_y)
        size = data_x.nbytes * 5  # pre-defined a memory of size 5 times bigger than actual size

        dim = len(data_x.shape)
        channels = 1 if dim < 4 else data_x.shape[-3]
        height = 1 if dim < 3 else data_x.shape[-2]
        width = 1 if dim < 2 else data_x.shape[-1]

        env = lmdb.open(os.path.join(os.getcwd(), fileName), map_size=size)
        with env.begin(write=True) as txn:
            # txn is a Transaction object
            for i in range(n_samples):
                datum = caffe.proto.caffe_pb2.Datum()
                datum.channels = channels
                datum.height = height
                datum.width = width
                print(data_x[i])
                datum.data = data_x[i].tostring()  # or .tostring() if numpy < 1.9
                datum.label = data_y[i]
                str_id = '{:08}'.format(i)

                # The encode is only essential in Python 3
                txn.put(str_id.encode('ascii'), datum.SerializeToString())

