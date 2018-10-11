# LivenessDectectionForFaceSpoofingAttack

### This is a project designed to handle face spoofing attack in the face recognition camera. (on going)

This project is designed to distinguish between real person and fake images or videos shows up in front of the security camera.

This project contains a SVM models trained on Rotation Invariant Uniform LBP histogram. The historgram vector is extraced from the DOG(difference of gaussian iamge) on all channels of HSV and YCbCr colorspaces. 

#### Usage:

```
$ cd {project root}/src
$ make
$ cd {project root}/bin    
$ ./main
```
Then you should see four data files and one model file. Accuracy and detial will show in the command window. 

#### Note:
  1. Currently, test and train are integrated in main.cpp
  2. Current version performs DOG on all colorspace channels efore LBP feature extraction.
  3. Replacing SVM with a MLP model is currently in test. (future option)

---------------------
update 10/09/2018:
  1. Optimized the baisc LBP image calculation.
  2. Add wrapper for all 3 modes of LBP feature extraction and reorganized the class structure.
  
update 10/11/2018:
  1. Added difference of gaussian on each colorspace channel before the LBP histogram feature vector extraction. Now the accuracy is 87.47% on the current dataset.
  2. Current configuration has image resize to 96x96 and LBP cellsize of 16.
