# Anti Face Spoofing Attack Based On Facial LBP and Optical Flow

### This project is designed to handle face spoofing attack in the face recognition camera. (updating)

This project is designed to distinguish between real person and fake images or videos shows up in front of the security camera.

The current algorithm uses LBP extraction on the facial image and the Optical Flow image(adding a gray scale lbp in still in test). Multiple MLP networks are trained on their independent dataset, then their own intermediate encoded results will be  combined to generate a merged sample vector for traning a merging MLP model, where the final prediction is generated.  

--------------------------
#### Usage:
##### C++ program:
dependency: opencv(with HDF5 module support, dnn module, ml module)

###### There will be multiple executable files generated after make the source code.

###### sperate data preparation: 
dog_lbp, ofm, [gray, still in grogress]:

These seperate MLP models need to be trained independently on three dataset. Before training the MLP, train and test data<data, labels> needs to be formated into HDF5 files. 
For exapmle: "dog_lbp" generates the "train.h5" and "test.h5"(name can be user specified) stores the train and test data, and then trains a SVM model then output the SVM model accuracy just for comparison. The generated database files are used to independently train the DOG_LBP_MLP model.
(The generated HDF5 files can be used to generate a MLP model by python in the /src/python/MLP folder)
A sample program generates the dog_lbp dataset file is as follow:
```
$ cd {project root}/src
$ make [default to make all, other options: make lbp, make ofm etc...]
$ cd {project root}/bin    
$ ./lbp -d <path-to-the-data-root-directory> [OPTIONAL: -r <resize image size>, -c <cell size for lbp feature extraction>] 
```
###### integration
integration: loads three MLP networks trained on the LBP and Optical Flow feature [and Gray scale image], then run the inferece on individual input image to generate a "intermediate result", then stores the result into another "train.h5" and "test.h5"(name is also user spefified) files. These two files are used to trained the "merging network" (MLP as well). Training and testing code are also in the /src/python/MLP folder.
```
$ ./integration -m1 path-to-model1 -b1 path-to-model1-parameter -m2 path-to-model2 -b2 path-to-model2-parameter -m3 path-to-model3 -b3 path-to-model3-parameter -d1 path-to-data-folder -d2 path-to-data-folder -d3 path-to-data-folder -r image-resize-factor -c lbp-cell-size
```
[note: data root directory should contain two folders: train and test.]
Then you should see generated data files and one model file. Accuracy and detial will show in the command window. 

##### Python program:
dependency: python-opencv, h5py, caffe

Source code in the MLP folder are tweakable generic program to build, train and test user defined model in Caffe. The model configuration are in MLP.py and the main dirver program (for train and test and calls model construction method) is in main.py.
A example usage is as follow:
```
$ cd {project root}/src/python/MLP
$ python main.py -e <number of desired epochs> 
```
This will train (test while traininig) a MLP model on the same HDF5 data generated in the /bin directory by the C++ program. User can specify their own data by using other input arguments. See detail usage in main.py
[note: Please be carefull that the train and test data format has to match the MLP model definition, which is configured in the MLP.py file]

For testing the model:

A simple inference program is inference.py, which loads the exsisting pre-trained model and test the model accuracy. User can tweak with it to do their own testing. 
```
$ python inference.py -m <model path> -c<checkpoint file path> OPTIONAL[-d <path to the hdf5 file>]
```
This will run the test on the whole data set (default 'test.h5' in /bin) after the MLP model training finished.


#### Note:
  1. Currently, the merging MLP is still in development.
  2. Current version lbp performs DOG on all colorspace channels followed by LBP feature extraction on all channels.
  3. sub MLP network trained solely on LBP feature can obtain an (best) accuracy around 97% bu it self.
  4. This project does not generate OFM (optical flow map) data, but directly used it to train a sub MLP network.
  5. All train and test data need to be prepared seperatly for each sub MLP model, along with the pre-trained MLP binary to generate the dataset for training the merging MLP model. 

---------------------
update 10/09/2018:
  1. Optimized the baisc LBP image calculation.
  2. Add wrapper for all 3 modes of LBP feature extraction and reorganized the class structure.
  
update 10/11/2018:
  1. Added difference of gaussian on each colorspace channel before the LBP histogram feature vector extraction. Now the accuracy is 87.47% on the current dataset.
  2. Current configuration has image resize to 96x96 and LBP cellsize of 16.

update 10/12/2018:
  1. Fixed bug in calculating uniform LBP pattern.
  2. Added test only option in main.cpp
  
update 10/15/2018:
  1. Added option for export processed train/test set into txt files.
  2. Added generic python util for generating autoencoder on given feature vectors.
  3. Start working on generic python MLP model generation. 

update 10/17/2018:
  1. Now data can be written into HDF5 file directly after processing.
  2. Python MLP training can directly proceed from the C++ generated HDF5 file.

update 10/18/2018:
  1. generated a deploy model for single inference purpose. 
  2. added a inference script to run the test on whole test set after training the MLP.

update 10/24/2018:
  1. added another executable for generating data for training the merging MLP
  2. update the Makefile

update 10/29/2018:
  1. change the structure in Data.cpp. Now Data is a generic data preparation object thats can be told to prepare the data in a user specified routine. It accepts a variadic function pointer whos input argument can vary by function. 
  2. Starting the integration.cpp
  
update 10/30/2018:
  1. Now integrating.cpp can loads multiple models and do the inference seperately to prepare the merged feature vector for training the merging MLP model. 
