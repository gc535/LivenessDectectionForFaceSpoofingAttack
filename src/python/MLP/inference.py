import caffe
import os

### parse argument
ag = argparse.ArgumentParser()
ag.add_argument("-m", "--model-path", required=True, help="path to the model definition .prototxt file")
ag.add_argument("-c", "--checkpoint-path", required=True, help="path to the test model checkpoint file .caffemodel")
ag.add_argument("-d", "--data-path", required=False, help="path to the hdf5 data file")    
args = vars(ag.parse_args())

if os.path.exists(args["model_path"]):
    model_path = args["model_path"]
else:
    print("[Error]: Model definition dose not exist. Aborted.")
    exit(1)

if os.path.exists(args["checkpoint_path"]):
    checkpoint_path = args["checkpoint_path"]
else:
    print("[Error]: Model checkpoint file dose not exist. Aborted.")
    exit(1)


data_path = os.path.join(os.getcwd(), '../../../bin/test.h5')
if os.path.exists(args["data_path"]):
    data_path = args["data_path"]
elif os.path.exists(args["data_path"]):
    print("no data file given, using default in /bin directory")
else:
    print("[Error]: Test file dose not exist. Aborted.")
    exit(1)
test_file = h5py.File(data_path, 'r')
test_data_len = len(test_file['data'])


### load model 
net = caffe.Net(model_path,
                checkpoint_path,
                caffe.TEST) 

### start testing
# test whole test set
print("[Note]: Start testing...")
fake_cnt, living_cnt = 0, 0
fake_total, living_total = 0, 0
for itr in range(test_data_len//100):   # hard code the batchsize here, since it has to be the same as in the *_test.protxt file
    net.forward()
    result = net.blobs['prob'].data.argmax(1)
    expected_label = net.blobs['label'].data.squeeze()
    for sample in range(len(result)):
        # count actaul fake/living numbers 
        if expected_label[sample] == 1: 
            living_total += 1
        else:
            fake_total += 1
        # count predicted fake/living numbers    
        if result[sample] == 1:
            living_cnt += 1
        else:
            fake_cnt += 1
print("total living samples: ", living_total, " total fake samples: ", fake_total)
print("total living detected: ", living_cnt, " total fake detected: ", fake_cnt)