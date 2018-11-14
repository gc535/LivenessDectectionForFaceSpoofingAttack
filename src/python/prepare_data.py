# MIT License
# 
# Copyright (c) 2016 David Sandberg
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

#
# Borrowed from davidsandberg's facenet project: https://github.com/davidsandberg/facenet
# From this directory:
#   facenet/src/align
#
# Just keep the MTCNN related stuff and removed other codes
# python package required:
#     tensorflow, opencv,numpy


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import argparse
import numpy as np
#import cv2

from random import seed
from random import random

def prepare_data(input_dir, output_dir):
   
    train_list = os.path.join(output_dir,"train_image_list.txt")
    test_list = os.path.join(output_dir,"test_image_list.txt") 

    train_file = open(train_list,'w'); 
    test_file = open(test_list,'w');  

    seed(999) 

    for root, dirs, files in os.walk(input_dir, topdown=False):
        for subdir in dirs: 
             src_dir = os.path.join(root, subdir)
             print(src_dir) 

             filelist = (f for f in os.listdir(src_dir) if ( f.endswith('.' + 'bmp') or f.endswith('.' + 'jpg') or f.endswith('.' + 'png') ) )  
             #print(filelist) 

             for f in filelist: 
                 #print(f) 
                 filename =  os.path.join(os.path.abspath(src_dir),f)
                 print('Process '+filename); 

                 #if random()<0.1: 
                 if 'train' in filename: 
                     train_file.write(filename+'\n');
                 elif 'test' in filename: 
                     test_file.write(filename+'\n');
                 else:
                     continue   


    train_file.close()
    test_file.close()             

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='image to be detected for faces.',default='./')
    parser.add_argument('--output', type=str, help='new image with boxed faces',default='./')
    return parser.parse_args(argv)

if __name__ == '__main__':
    args = (parse_arguments(sys.argv[1:])) 
    prepare_data(args.input,args.output) 
     

