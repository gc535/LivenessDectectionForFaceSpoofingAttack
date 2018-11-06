from caffe import layers as L
from caffe import params as P
from caffe.proto import caffe_pb2
import caffe
import os


def createAutoencoder(hdf5, input_size, batch_size, phase):
    n = caffe.NetSpec()
    if phase == "inference":
        n.data = L.Input(input_param={'shape': {'dim':[1,input_size]}})
    else:    
        n.data = L.HDF5Data(batch_size=batch_size, source=hdf5, ntop=1)
    n.ip1 = L.InnerProduct(n.data, num_output=256, weight_filler=dict(type='xavier'))
    n.bottleneck = L.Sigmoid(n.ip1, in_place=True)
    n.decode = L.InnerProduct(n.bottleneck, num_output=input_size, weight_filler=dict(type='xavier'))
    n.loss = L.EuclideanLoss(n.decode, n.data)

    
    return n.to_proto()

def Solver(model_name, trainModel, testModel, total_samples, train_batch_size, epochs, outputDir=None):
    s = caffe_pb2.SolverParameter()
    s.solver_mode: GPU
    s.random_seed = 0xCAFFE

    # Specify locations of the train and (maybe) test networks.
    s.train_net = trainModel
    #s.max_iter = 50     # no. of times to update the net (training iterations)

    s.test_net.append(testModel)
    s.test_interval = 500  # Test after every 500 training iterations.
    s.test_iter.append(300) # Test on 10 batches each time we test.

    # EDIT HERE to try different solvers
    # solver types include "SGD", "Adam", and "Nesterov" among others.
    s.type = "Adam"

    # Set the initial learning rate for SGD.
    s.base_lr = 0.00001  # EDIT HERE to try different learning rates      #current best shot: 0.001
    # Set momentum to accelerate learning by
    # taking weighted average of current and previous updates.
    s.momentum = 0.90
    # Set weight decay to regularize and prevent overfitting
    s.weight_decay = 5e-4

    # Set `lr_policy` to define how the learning rate changes during training.
    # This is the same policy as our default LeNet.
    s.lr_policy = 'inv'
    s.gamma = 0.0001                                                   #current best shot: 0.00001
    s.power = 0.75
    # EDIT HERE to try the fixed rate (and compare with adaptive solvers)
    # `fixed` is the simplest policy that keeps the learning rate constant.
    #s.lr_policy = 'fixed'

    # Display the current training loss and accuracy every 1000 iterations.
    s.display = 1000

    # Snapshots are files used to store networks we've trained.
    # We'll snapshot every 5K iterations -- twice during training.
    s.snapshot = epochs*(total_samples//train_batch_size) - 1
    s.snapshot_prefix = model_name

    # Train on the GPU
    #s.solver_mode = caffe_pb2.SolverParameter.CPU

    # Write the solver to a temporary file and return its filename.
    output_dir = os.getcwd()
    if outputDir:
        output_dir = outputDir
    solver_path = os.path.join(output_dir, 'solver.prototxt')
    with open(solver_path, 'w') as f:
        f.write(str(s))

    return solver_path
