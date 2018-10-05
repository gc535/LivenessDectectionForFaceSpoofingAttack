# LivenessDectectionForFaceSpoofingAttack

### This is a project designed to handle face spoofing attack in the face recognition camera. (on going)

This project utilized Rotation Invariant Uniform LBP pattern in different color spaces to distinguish between real person and fake images or videos. Currently, prediction is made by trained SVM model taking rotation invarient, uniform, normalized LBP feature vector. 

#### Usage:

```
$ cd {project root}/src
$ make
$ cd {project root}/bin    
$ ./main
```
Then you should see four data files and one model file. Accuracy and detial will show in the command window. 

Note:
  1. Currently, test and train are integrated in main.cpp
  2. Current version does not perform DOG on image in both color spaces. (future option)
