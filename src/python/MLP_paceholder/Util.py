import os
import sys
from matplotlib import pyplot as plt

### monitor object plots the traning curve ###
class Monitor(object):
    def __init__(self):
        self.i = 0
        self.x = []
        self.accuracy = []
        self.losses = []
        self.fig = plt.figure()
        plt.ion()
        
        
    def update(self, loss, accuracy):
        self.x.append(self.i)
        self.losses.append(loss)
        self.accuracy.append(accuracy)
        self.i += 1

        plt.gcf().clear()
        plt.plot(self.x, self.losses, label="loss")
        plt.plot(self.x, self.accuracy, label="accuracy")
        plt.legend()
        plt.draw()
        plt.pause(0.001)

######## util to load all images from current directory ##########
def listimages(img_dir):
    if not os.path.exists(img_dir):
        raise Exception("image dir is not exists")
    files = []
    if os.path.isfile(img_dir):
        files.append(img_dir)
    else:
        files = os.listdir(img_dir)

    if len(files) < 1:
        raise Exception("No picture needs to be detected")
    return files


######### Util method for percentage bar display #######
def progress_bar_util(action):
    cur_dir = os.getcwd()
    par_dir = os.path.join(cur_dir, os.pardir)
    toolbar_width = 40
    TOTAL_NUM_FILES = 0
    for label in range(4):    # directory idx used as tag
        gallery_path = os.path.join(par_dir, action, str(label))
        imgFiles = listimages(gallery_path)
        TOTAL_NUM_FILES += len(imgFiles)
    sys.stdout.write("preparing %sing data...\n" % action)
    sys.stdout.write("[%s]" % (" " * toolbar_width))
    sys.stdout.flush()
    sys.stdout.write("\b" * (toolbar_width+1)) # return to start of line, after '['
    FILE_PER_PERCENT = TOTAL_NUM_FILES / 40.0 
    progress_tracker = 0

    return (progress_tracker, FILE_PER_PERCENT)

def test_progress_bar_util(action, label):
    cur_dir = os.getcwd()
    par_dir = os.path.join(cur_dir, os.pardir)
    toolbar_width = 40
    gallery_path = os.path.join(par_dir, action, str(label))
    imgFiles = listimages(gallery_path)
    TOTAL_NUM_FILES = len(imgFiles)
    sys.stdout.write("\npreparing %sing data...\n" % action)
    sys.stdout.write("[%s]" % (" " * toolbar_width))
    sys.stdout.flush()
    sys.stdout.write("\b" * (toolbar_width+1)) # return to start of line, after '['
    FILE_PER_PERCENT = TOTAL_NUM_FILES / 40.0
    progress_tracker = 0
    return (progress_tracker, FILE_PER_PERCENT)


