/*
  Copyright (C) 2017 Open Intelligent Machines Co.,Ltd

  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
*/

#include <iostream>

#include <string>
#include <vector>
#include <cmath>

#include "Util.hpp"

using namespace std;
using namespace cv;

#include "shear.hpp" 

#define PI 3.14159265

int save_hdf5= 1; 
static int dir_count = 0;


void atrousfilters(cv::Mat &h0Out, cv::Mat &h1Out,  cv::Mat &g0Out,  cv::Mat &g1Out, char *mode) 
{

 // Quasi-tight frame filters, frame bounds A=.97 and B=1

 double  h0_0[7][7]= { {-7.900496718847182e-07, 0., 0.000014220894093924927, 0.000025281589500310983, -0.000049773129328737247, -0.00022753430550279883, -0.00033182086219158167}, 
                    { 0,               0,              0,                   0,                       0,                          0,                     0},
                    { 0.000014220894093924927, 0., -0.0002559760936906487, -0.00045506861100559767,   0.0008959163279172705,   0.004095617499050379,    0.00597277551944847},
                    { 0.000025281589500310983, 0., -0.00045506861100559767, 0.0009765625,            0.0015927401385195919,  -0.0087890625,          -0.01795090623402861},
                    { -0.000049773129328737247, 0.,  0.0008959163279172705,  0.0015927401385195919,   -0.0031357071477104465,  -0.014334661246676327,   -0.020904714318069645},
                    { -0.00022753430550279883,  0.,  0.004095617499050379,  -0.0087890625,            -0.014334661246676327,    0.0791015625,            0.16155815610625748},
                    { -0.00033182086219158167,  0.,  0.00597277551944847,   -0.01795090623402861,     -0.020904714318069645,    0.16155815610625748,     0.3177420190660832} };

 double g0_0[10][10] = { { -6.391587676622346e-010,             0.,                1.7257286726880333e-08,    3.067962084778726e-08,   -1.3805829381504267e-07,  -5.522331752601707e-07,  -3.3747582932565985e-07,    1.9328161134105974e-06,     5.6949046198705095e-06,    7.649452131381623e-06},
                      {  0.,                          0.,                      0. ,                         0.,               0.,  0.,  0.,  0.,  0.,  0. },
                      { 1.7257286726880333e-08,             0.,               -4.65946741625769e-07,     -8.283497628902559e-07,    3.727573933006152e-06,    0.000014910295732024608,     9.111847391792816e-06,    -0.000052186035062086126,   -0.00015376242473650378,   -0.00020653520754730382},
                      { 3.067962084778726e-08,              0.,               -8.283497628902559e-07,    -1.2809236054493144e-06,   6.6267981031220475e-06,   0.00002305662489808766,     0.000010064497559808503,  -0.0000806981871433068,     -0.00021814634152337594,   -0.00028666046030363884},
                      {-1.3805829381504267e-07,             0.,                3.727573933006152e-06,     6.6267981031220475e-06,  -0.000029820591464049215, -0.00011928236585619686,    -0.00007289477913434253,    0.000417488280496689,       0.0012300993978920302,     0.0016522816603784306},
                      {-5.522331752601707e-07,              0.,                0.000014910295732024608,   0.00002305662489808766,  -0.00011928236585619686,  -0.00041501924816557786,    -0.00018116095607655303,    0.0014525673685795225,      0.0039266341474207675,     0.005159888285465499},
                      {-3.3747582932565985e-07,             0.,                9.111847391792816e-06,     0.000010064497559808503, -0.00007289477913434253, -0.00018116095607655303,     0.001468581806076247,      0.0006340633462679356,     -0.01181401175635013,      -0.021745034491193898},
                      {1.9328161134105974e-06,             0.,               -0.000052186035062086126,  -0.0000806981871433068,    0.000417488280496689,    0.0014525673685795225,       0.0006340633462679356,    -0.005083985790028328,      -0.013743219515972684,     -0.018059608999129246},
                      {5.6949046198705095e-06,             0.,               -0.00015376242473650378,   -0.00021814634152337594,   0.0012300993978920302,   0.0039266341474207675,       -0.01181401175635013,      -0.013743219515972684,       0.0826466923977296,        0.1638988884584603},
                      {7.649452131381623e-06,              0.,               -0.00020653520754730382,   -0.00028666046030363884,   0.0016522816603784306,   0.005159888285465499,        -0.021745034491193898,     -0.018059608999129246,       0.1638988884584603,        0.31358726209239235}  };
  double g1_0[7][7] = {  {-7.900496718847182e-07,    0.,      0.000014220894093924927, 0.000025281589500310983, -0.000049773129328737247, -0.00022753430550279883, -0.00033182086219158167}, 
                      { 0,                 0,                0,                0,                            0,                       0,                       0},
                      { 0.000014220894093924927,  0.,     -0.0002559760936906487,  -0.00045506861100559767,   0.0008959163279172705,    0.004095617499050379,    0.00597277551944847},
                      { 0.000025281589500310983,  0.,     -0.00045506861100559767, -0.0009765625,             0.0015927401385195919,    0.0087890625,            0.01329909376597139},
                      {-0.000049773129328737247,  0.,      0.0008959163279172705,   0.0015927401385195919,   -0.0031357071477104465,   -0.014334661246676327,   -0.020904714318069645},
                      {-0.00022753430550279883,   0.,      0.004095617499050379,    0.0087890625,            -0.014334661246676327,    -0.0791015625 ,          -0.1196918438937425  },
                      {-0.00033182086219158167,   0.,      0.00597277551944847,     0.01329909376597139,     -0.020904714318069645,    -0.1196918438937425,      0.8177420190660831  } };

  double h1_0[10][10] = { {6.391587676622346e-010,       0.,                 -1.7257286726880333e-08,          -3.067962084778726e-08,     1.3805829381504267e-07,  5.522331752601707e-07,     3.3747582932565985e-07, -1.9328161134105974e-06,  -5.6949046198705095e-06,          -7.649452131381623e-06}, 
                       {0.,                     0.,                           0.,                            0.,                      0.,                      0., 0., 0., 0., 0.},   
                       {-1.7257286726880333e-08,       0.,                  4.65946741625769e-07,             8.283497628902559e-07,    -3.727573933006152e-06,   -0.000014910295732024608,    -9.111847391792816e-06,   0.000052186035062086126,  0.00015376242473650378,           0.00020653520754730382},
                       {-3.067962084778726e-08,        0.,                  8.283497628902559e-07,           -2.9917573832012203e-07,   -6.6267981031220475e-06,   5.3851632897621965e-06,     0.00004049868144081346, -0.00001884807151416769,  -0.00023692226948222173 ,         -0.0003769812640795245},
                       {1.3805829381504267e-07,       0.,                 -3.727573933006152e-06,           -6.6267981031220475e-06,    0.000029820591464049215,  0.00011928236585619686,     0.00007289477913434253, -0.000417488280496689,    -0.0012300993978920302 ,          -0.0016522816603784306},  
                       {5.522331752601707e-07,        0.,                 -0.000014910295732024608,          5.3851632897621965e-06,    0.00011928236585619686,  -0.00009693293921571956,    -0.0007289762659346422,   0.00033926528725501844 ,  0.004264600850679991 ,            0.006785662753431441},
                       {3.3747582932565985e-07,       0.,                 -9.111847391792816e-06,            0.00004049868144081346,    0.00007289477913434253,  -0.0007289762659346422,    -0.001468581806076247,    0.002551416930771248,     0.01181401175635013 ,             0.017093222023136675},
                       {-1.9328161134105974e-06,       0.,                  0.000052186035062086126 ,        -0.00001884807151416769 ,  -0.000417488280496689 ,    0.00033926528725501844,     0.002551416930771248,   -0.0011874285053925643,   -0.01492610297737997,             -0.023749819637010044}, 
                       {-5.6949046198705095e-06,       0.,                  0.00015376242473650378,          -0.00023692226948222173,   -0.0012300993978920302,    0.004264600850679991,     0.01181401175635013,    -0.01492610297737997,     -0.0826466923977296,              -0.12203257624594532},
                       {-7.649452131381623e-06,        0.,                  0.00020653520754730382,          -0.0003769812640795245,    -0.0016522816603784306,    0.006785662753431441,     0.017093222023136675,   -0.023749819637010044,    -0.12203257624594532,               0.821896776039774}  };
  
  // Matlab  h0 = [h0   fliplr(h0(:,1:end-1))];
  cv::Mat h0 = cv::Mat(7,7, CV_64FC1, h0_0); 
  cv::Mat h0_1 = h0.colRange(0,h0.cols-1); 
  cv::Mat h0_1_fliplr; 
  cv::flip(h0_1, h0_1_fliplr, 1);  
  cv::Mat h0_2 = mergeCols(h0,h0_1_fliplr); 
  cv::Mat h0_3 = h0_2.rowRange(0,h0_2.rows-1); 
  cv::Mat h0_3_flipud; 
  cv::flip(h0_3, h0_3_flipud, 0); 
  cv::Mat h0_4 = mergeRows(h0_2,h0_3_flipud); 
  h0Out = h0_4.clone(); 

  // Matlab  g0 = [g0   fliplr(g0(:,1:end-1))];
  cv::Mat g0 = cv::Mat(10,10, CV_64FC1, g0_0); 
  cv::Mat g0_1 = g0.colRange(0,g0.cols-1); 
  cv::Mat g0_1_fliplr; 
  cv::flip(g0_1, g0_1_fliplr, 1);  
  cv::Mat g0_2 = mergeCols(g0,g0_1_fliplr); 
  cv::Mat g0_3 = g0_2.rowRange(0,g0_2.rows-1); 
  cv::Mat g0_3_flipud; 
  cv::flip(g0_3, g0_3_flipud, 0); 
  cv::Mat g0_4 = mergeRows(g0_2,g0_3_flipud); 
  g0Out = g0_4.clone(); 

  // Matlab  g1 = [g1   fliplr(g1(:,1:end-1))];
  cv::Mat g1 = cv::Mat (7,7, CV_64FC1, g1_0); 
  cv::Mat g1_1 = g1.colRange(0,g1.cols-1); 
  cv::Mat g1_1_fliplr; 
  cv::flip(g1_1, g1_1_fliplr, 1);  
  cv::Mat g1_2 = mergeCols(g1,g1_1_fliplr); 
  cv::Mat g1_3 = g1_2.rowRange(0,g1_2.rows-1); 
  cv::Mat g1_3_flipud; 
  cv::flip(g1_3, g1_3_flipud, 0); 
  cv::Mat g1_4 = mergeRows(g1_2,g1_3_flipud); 
  g1Out = g1_4.clone(); 

  // Matlab  h1 = [h1   fliplr(h1(:,1:end-1))];
  cv::Mat h1 = cv::Mat(10,10, CV_64FC1, h1_0); 


  cv::Mat h1_1 = h1.colRange(0,h1.cols-1); 
  cv::Mat h1_1_fliplr; 
  cv::flip(h1_1, h1_1_fliplr, 1);  
  cv::Mat h1_2 = mergeCols(h1,h1_1_fliplr); 



  cv::Mat h1_3 = h1_2.rowRange(0,h1_2.rows-1); 
  cv::Mat h1_3_flipud; 
  cv::flip(h1_3, h1_3_flipud, 0); 


  cv::Mat h1_4 = mergeRows(h1_2,h1_3_flipud); 
  h1Out = h1_4.clone(); 

}  

/*

function yT=symext(x,h,shift);

% FUNCTION Y = SYMEXT
% INPUT:  x, mxn image
%         h, 2-D filter coefficients
%         shift, optional shift
% OUTPUT: yT image symetrically extended (H/V symmetry)
%
% Performs symmetric extension for image x, filter h. 
% The filter h is assumed have odd dimensions.
% If the filter has horizontal and vertical symmetry, then 
% the nonsymmetric part of conv2(h,x) has the same size of x.
%
% Created by A. Cunha, Fall 2003;
% Modified 12/2005 by A. Cunha. Fixed a bug on wrongly 
% swapped indices (m and n). 

[m,n] = size(x);
[p,q] = size(h);
parp  = 1-mod(p,2) ;
parq  = 1-mod(q,2);

p2=floor(p/2);q2=floor(q/2);
s1=shift(1);s2=shift(2);

ss = p2 - s1 + 1;
rr = q2 - s2 + 1;

yT = [fliplr(x(:,1:ss)) x  x(:,n  :-1: n-p-s1+1)];
yT = [flipud(yT(1:rr,:)); yT ;  yT(m  :-1: m-q-s2+1,:)];
yT = yT(1:m+p-1 ,1:n+q-1);
*/


cv::Mat symext(cv::Mat &img, cv::Mat h, int s1, int s2) 
{
    int m = img.rows;
    int n = img.cols; 
    int p = h.rows; 
    int q = h.cols; 

    int parp = 1 - p%2; 
    int parq = 1 - q%2; 

    int p2 = floor(0.5*p); 
    int q2 = floor(0.5*q); 

    int ss = p2-s1+1; 
    int rr = q2-s2+1; 
  
    //cout << "parp = " << parp << " parq = " << parq << " p2 = " << p2 << " q2 = " << q2 << " ss = " << ss << " rr " << rr << endl; 

    //yT = [fliplr(x(:,1:ss)) x  x(:,n  :-1: n-p-s1+1)];
    cv::Mat img_ss = img.colRange(0,ss);   
    cv::Mat img_ss_flip; 
    cv::flip(img_ss ,img_ss_flip,1);  


    cv::Mat img_ss_rev; 
    cv::flip(img.colRange(n-p-s1,img.cols), img_ss_rev,1) ; //x(:,n  :-1: n-p-s1+1)

    cv::Mat yT1; 
    yT1 = mergeCols(img_ss_flip,img); 
    yT1 = mergeCols(yT1,img_ss_rev); 

    //yT = [flipud(yT(1:rr,:)); yT ;  yT(m  :-1: m-q-s2+1,:)];
    cv::Mat yT2_1;  
    cv::flip(yT1.rowRange(0,rr), yT2_1,0) ; //x(:,n  :-1: n-p-s1+1)
    cv::Mat yT2_2;
    cv::flip(yT1.rowRange(m-q-s2,m), yT2_2,0); 
    cv::Mat yT2 = mergeRows(yT2_1,yT1); 
    yT2 = mergeRows(yT2,yT2_2); 
 
    //yT = yT(1:m+p-1 ,1:n+q-1);
    yT2 =   yT2.rowRange(0,m+p-1); 
    yT2 =   yT2.colRange(0,n+q-1); 
 
    return yT2; 
}

//2D convolution  
void conv2(const Mat &img, const Mat& kernel, ConvolutionType type, Mat& dest) {
  Mat source = img;
  if(CONVOLUTION_FULL == type) {
    source = Mat();
    const int additionalRows = kernel.rows-1, additionalCols = kernel.cols-1;
    copyMakeBorder(img, source, (additionalRows+1)/2, additionalRows/2, (additionalCols+1)/2, additionalCols/2, BORDER_CONSTANT, Scalar(0));
  }
 
  Point anchor(kernel.cols - kernel.cols/2 - 1, kernel.rows - kernel.rows/2 - 1);
  int borderMode = BORDER_CONSTANT;
  Mat kernel_flip;  
  flip(kernel,kernel_flip,1); 
 
  filter2D(source, dest, img.depth(), kernel_flip, anchor, 0, borderMode);
 
  if(CONVOLUTION_VALID == type) {
    dest = dest.colRange((kernel.cols-1)/2, dest.cols - kernel.cols/2)
               .rowRange((kernel.rows-1)/2, dest.rows - kernel.rows/2);
  }
}

cv::Mat upsample2df(cv::Mat &h, int power) 
{
   int rate = pow(2,power); 
   cv::Mat hout = cv::Mat(h.rows*rate,h.cols*rate,CV_64FC1,cv::Scalar(0)); 
   for(int r=0; r<h.rows;r++) 
   {
      for(int c=0; c<h.cols;c++) 
      {
         hout.at<double>(r*rate,c*rate) = h.at<double>(r,c);
      }
   }
   return hout;      
}


/******************************************************************
* atrousc.c -  Written by Arthur Cunha. This routine builds up on 
*               zconv2D_OS.c written by Jason Laska
*
* Inputs:   x - A 2D signal
*           h - 2D filter
*           m - separable upsampling matrix
*         
* Outputs:  y - 2D result of convolution with filter 
*           upsampled by a m, only the 'valid' part is returned.
*           Similar to conv2(x,h,'valid'), where h is the upsampled
*           filter.
*  
*          
*
* Usage:    y = zconv2D_O(x,h,m);
*
* Notes:    This function does not actually upsample the filter, 
*           it computes the convolution as if the filter had been 
*           upsampled. This is the ultimate optimized version.
*           Further optimized for separable (diagonal) upsampling matrices.
*
* This is a MEX-FILE for matlab
*
/********************************************************/

cv::Mat  atrousc(cv::Mat &x, cv::Mat &h, cv::Mat &I2, int L)
{

   //double *FArray,*SArray,*outArray,*M;
/* FArray   - Filter coefficients
   SArray   - Signal coefficients
   outArray - Output coefficients
   M        - upsampling matrix 	*/

   int SColLength,SRowLength,FColLength,FRowLength,Out_SColLength,Out_SRowLength;
   int SFColLength,SFRowLength;
   int n1,n2,l1,l2,k1,k2,f1,f2, kk2, kk1;
   double sum;   
   int M0,M3,sM0,sM3;


    SColLength = x.rows;  
    SRowLength = x.cols;
    FColLength = h.rows;  
    FRowLength = h.cols;
    
    SFColLength = FColLength-1;
    SFRowLength = FRowLength-1;

    M0 = (int) I2.at<double>(0,0)*L;    
    M3 = (int) I2.at<double>(1,1)*L;  
    sM0 = M0-1;
    sM3 = M3-1;
    
    Out_SColLength = SColLength - M0*FColLength + 1;
    Out_SRowLength = SRowLength - M3*FRowLength + 1;
	
    cv::Mat OUT = cv::Mat(Out_SColLength, Out_SRowLength, CV_64FC1);  

    /* Convoluyion loop */

    for (n1=0;n1<Out_SRowLength;n1++){
	for (n2=0;n2<Out_SColLength;n2++){
            sum=0;		    
	    kk1 = n1 + sM0;
	    for (k1=0;k1<FRowLength;k1++){
  	       kk2 = n2 + sM3;
	       for (k2=0;k2<FColLength;k2++){
		 f1 = SFRowLength - k1; /* flipped index */
		 f2 = SFColLength - k2;  		
	         sum+= h.at<double>(f2,f1) * x.at<double>(kk2,kk1);					
		 kk2+=M3;
	       }
	       kk1+=M0;
	    } 
	    OUT.at<double>(n2,n1)= sum;
	}
    }

    return OUT;
}


void printDebugFile(cv::Mat &xx, string fn)
{
   cv::Mat x2 = xx.clone(); 
   xx.convertTo(x2,CV_64FC1); 
   FILE *fp = fopen(fn.c_str(),"wt"); 
   for(int r=0; r<x2.rows;r++) {
      for (int c=0; c<x2.cols; c++) {
          fprintf(fp,"%f ", x2.at<double>(r,c)); 
      }
      fprintf(fp,"\n");
   }
   fclose(fp);
}

void  atrousdec(cv::Mat &x, cv::Mat &h0, cv::Mat &h1, int Nlevels,vector<cv::Mat> &y)
{
    cv::Mat se0 = symext(x, h0, 1, 1); 
    cv::Mat se1 = symext(x, h1, 1, 1); 
    se0.convertTo(se0,CV_64FC1); 
    se1.convertTo(se1,CV_64FC1); 

    cv::Mat y0;  
    cv::Mat y1; 

    conv2(se0,h0, CONVOLUTION_VALID,y0); 
    conv2(se1,h1, CONVOLUTION_VALID,y1); 


    //printDebugFile(y0, "y00.txt"); 
    //printDebugFile(y1, "y11.txt"); 

    y.insert(y.begin(),y1);  

    cv::Mat xx = y0; 
    cv::Mat I2 = cv::Mat(2,2,CV_64FC1); 
    I2.at<double>(0,0)=1;  I2.at<double>(1,1)=1;  
    I2.at<double>(0,1)=0;  I2.at<double>(1,0)=0;
    
    for(int i=1; i<=Nlevels-1; i++) 
    {
       int shift = -pow(2,i-1)+2; 
       int L = pow(2,i); 
       cv::Mat h0_up = upsample2df(h0,i); 
       cv::Mat h1_up = upsample2df(h1,i); 

       //printDebugFile(xx, "xx.txt"); 


       se0 = symext(xx, h0_up, shift,shift); 
       se0.convertTo(se0,CV_64FC1); 


       se1 = symext(xx, h1_up, shift,shift); 
       se1.convertTo(se1,CV_64FC1); 

       //printDebugFile(se0, "se0.txt"); 
       //printDebugFile(se1, "se1.txt"); 


       y0 = atrousc(se0,h0,I2,L);
       y1 = atrousc(se1,h1,I2,L);

   /*  cout << "se0 rows,cols = " << se0.rows << "," << se0.cols << endl; 
    cout << "se1 rows,cols = " << se0.rows << "," << se1.cols << endl; 
    cout << "y0 rows, cols = " << y0.rows << "," << y0.cols << endl; 
    cout << "y1 rows, cols = " << y1.rows << "," << y1.cols << endl; 

       printDebugFile(y0, "y0.txt"); 
       printDebugFile(y1, "y1.txt"); 
   */
       xx = y0; 
       y.insert(y.begin(),y1); 
 
       //cout << "************************************" << endl; 
       //fgetc(stdin); 

    }
    y.insert(y.begin(),xx); 

}

/*
function d = shear_trans(x,pfilt,shear)
% This function generates the shearlet coefficients.
%
% Input :
%        x : input image
%        pfilt: name of filter for nuosubsampled LP (see atrousdec.m)
%        shear: cell array of directional shearing filters
%               (see shearing_filters_Myer.m).
%
% Output :
%        d : cell array of shearlet coefficients
%
% For instruction, see instruction.txt.
%
%
% Written by Wang-Q Lim on May 5, 2010.
% Copyright 2010 by Wang-Q Lim. All Right Reserved.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/

void shear_transform(cv::Mat &img, cv::Mat &h0, cv::Mat &h1, vector<cv::Mat> &shear, vector<cv::Mat> &d) 
{
    int level = shear.size(); 
    vector<cv::Mat> y; 
    atrousdec(img,h0, h1,level,y); 
    
    d.push_back(y[0]); 

    vector<cv::Mat> y_fft_real; 
    vector<cv::Mat> y_fft_comp;     

/*
    FILE *fp = fopen("../y2.txt","rt"); 
    for(int r=0; r<256; r++) {
       for(int c=0;c<256; c++) {
         float tmp; 
         fscanf(fp,"%f ", &tmp); 
         y[1].at<double>(r,c) = tmp; 
       }
    }
    fclose(fp); */
 
    ifstream i1("../y2.txt"); 
    for(int r=0; r<256; r++) {
       for(int c=0;c<256; c++) {
         double tmp; 
         i1 >> tmp; 
         y[1].at<double>(r,c) = tmp; 
       }
    }
    i1.close(); 

    ofstream o1("y2_read.txt");
    o1 << y[1]<< endl;  
    o1.close(); 

    //printDebugFile(y[1], "y2_read.txt"); 


    // y[i]=fft2(y[i]);
    for (int i=1; i<y.size(); i++) 
    {    
        cv::Mat fftreal,fftcomp;
        cv::Mat junk;  
        fft2(y[1], junk, fftreal, fftcomp);  

        printDebugFile(fftreal, "fftreal.txt"); 
        printDebugFile(fftcomp, "fftcomp.txt"); 

        cout << "check fft2, i = " << i << "************************************" << endl; 
        fgetc(stdin); 

        y_fft_real.push_back(fftreal); 
        y_fft_comp.push_back(fftcomp); 
    }
    for(int j=1; j<=level; j++) 
    {
       int  num = 0; // number of shearlet direction 
       for (int k=1; k<=num; k++) 
       {
        //  d{j+1}(:,:,k) = (ifft2(shear{j}(:,:,k).*y{j+1}));     
       }
    }
}




// 2-D FFT. Make sure image size is power of 2
// srcImg is the real part of the input 
// imag is the imagnary part of input 

void fft2(cv::Mat &real, cv::Mat &imag, cv::Mat &fftreal, cv::Mat &fftcomplex, bool shift) 
{
    //cv::Mat real64;  
    //real.convertTo(real64, CV_64F); 
    assert(real.type()==CV_64F); 

    cv::Mat planes[2]; 
    planes[0] = real; 
    if(imag.empty()) 
        planes[1] = cv::Mat::zeros(real.size(), CV_64F); 
    else 
        planes[1] = imag; 

    cv::Mat Fourier;
    cv::merge(planes, 2, Fourier);
    cv::dft(Fourier, Fourier);
    // calcualte the dft respond magnitude
    split(Fourier, planes);

    fftreal = planes[0];
    fftcomplex  = planes[1];

    if(shift)  
    {
        // rearrange the quadrants of Fourier image  so that the origin is at the image center
        int cx = fftreal.cols/2;
        int cy = fftreal.rows/2;
        cv::Mat q0(fftreal, cv::Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
        cv::Mat q1(fftreal, cv::Rect(cx, 0, cx, cy));  // Top-Right
        cv::Mat q2(fftreal, cv::Rect(0, cy, cx, cy));  // Bottom-Left
        cv::Mat q3(fftreal, cv::Rect(cx, cy, cx, cy)); // Bottom-Right
        cv::Mat tmp;                           // swap quadrants (Top-Left with Bottom-Right)
        q0.copyTo(tmp);
        q3.copyTo(q0);
        tmp.copyTo(q3);
        q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
        q2.copyTo(q1);
        tmp.copyTo(q2);
        // rearrange the quadrants of Fourier image  so that the origin is at the image center
        q0 = cv::Mat(fftcomplex, cv::Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
        q1 = cv::Mat(fftcomplex, cv::Rect(cx, 0, cx, cy));  // Top-Right
        q2 = cv::Mat(fftcomplex, cv::Rect(0, cy, cx, cy));  // Bottom-Left
        q3 = cv::Mat(fftcomplex, cv::Rect(cx, cy, cx, cy)); // Bottom-Right
        // swap quadrants (Top-Left with Bottom-Right)
        q0.copyTo(tmp);
        q3.copyTo(q0);
        tmp.copyTo(q3);
        q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
        q2.copyTo(q1);
        tmp.copyTo(q2);
    }
}


// FFT on DOG Features
void fft2shift(cv::Mat &real, cv::Mat &imag, cv::Mat &fftreal, cv::Mat &fftcomplex)
{
    fft2(real, imag, fftreal, fftcomplex, true); 
} 


// 2-D inverse FFT 
void ifft2(cv::Mat &fftreal, cv::Mat &fftcomplex, cv::Mat &real, cv::Mat &imag, bool shift) 
{
    //first reverse the shift 
    if(shift)  
    {
        int cx = fftreal.cols/2;
        int cy = fftreal.rows/2;
        cv::Mat q0(fftreal, cv::Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
        cv::Mat q1(fftreal, cv::Rect(cx, 0, cx, cy));  // Top-Right
        cv::Mat q2(fftreal, cv::Rect(0, cy, cx, cy));  // Bottom-Left
        cv::Mat q3(fftreal, cv::Rect(cx, cy, cx, cy)); // Bottom-Right
        cv::Mat tmp;                           // swap quadrants (Top-Left with Bottom-Right)
        q0.copyTo(tmp);
        q3.copyTo(q0);
        tmp.copyTo(q3);
        q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
        q2.copyTo(q1);
        tmp.copyTo(q2);
        if(!fftcomplex.empty()) 
        {
            // crop the spectrum, if it has an odd number of rows or columns
            fftcomplex = fftcomplex(cv::Rect(0, 0, fftcomplex.cols & -2, fftcomplex.rows & -2));
            // rearrange the quadrants of Fourier image  so that the origin is at the image center
            q0 = cv::Mat(fftcomplex, cv::Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
            q1 = cv::Mat(fftcomplex, cv::Rect(cx, 0, cx, cy));  // Top-Right
            q2 = cv::Mat(fftcomplex, cv::Rect(0, cy, cx, cy));  // Bottom-Left
            q3 = cv::Mat(fftcomplex, cv::Rect(cx, cy, cx, cy)); // Bottom-Right
            // swap quadrants (Top-Left with Bottom-Right)
            q0.copyTo(tmp);
            q3.copyTo(q0);
            tmp.copyTo(q3);
            q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
            q2.copyTo(q1);
            tmp.copyTo(q2);
        }
    }

    cv::Mat planes[2];  
    planes[0] = fftreal;  
    if (fftcomplex.empty()) 
       planes[1] =  cv::Mat::zeros(fftreal.size(),CV_64F); 
    else 
       planes[1] = fftcomplex;   
    cv::Mat Fourier;  
    cv::merge(planes, 2, Fourier); 
    cv::dft(Fourier, Fourier, DFT_INVERSE + DFT_SCALE, 0);

    cv::split(Fourier, planes);
    real = planes[0];
    imag = planes[1]; 

}

// 2-D inverse FFT 
void ifft2shift(cv::Mat &fftreal, cv::Mat &fftcomplex, cv::Mat &real, cv::Mat &imag) 
{
    ifft2(fftreal, fftcomplex, real, imag, true);  

}


