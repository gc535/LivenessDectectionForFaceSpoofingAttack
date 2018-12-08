#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>

#include "Util.hpp"

using namespace std;

void atrousfilters(cv::Mat &h0Out, cv::Mat &h1Out,  cv::Mat &g0Out,  cv::Mat &g1Out, char *mode); 

cv::Mat symext(cv::Mat &img, cv::Mat h, int s1, int s2);

enum ConvolutionType {   
/* Return the full convolution, including border */
  CONVOLUTION_FULL, 
  
/* Return only the part that corresponds to the original image */
  CONVOLUTION_SAME,
  
/* Return only the submatrix containing elements that were not influenced by the border */
  CONVOLUTION_VALID
};

void conv2(const cv::Mat &img, const cv::Mat& kernel, ConvolutionType type, cv::Mat& dest) ;

cv::Mat upsample2df(cv::Mat &h, int power) ;

cv::Mat  atrousc(cv::Mat &x, cv::Mat &h, cv::Mat &I2, int L);

void printDebugFile(cv::Mat &xx, string fn);

void  atrousdec(cv::Mat &x, cv::Mat &h0, cv::Mat &h1, int Nlevels,vector<cv::Mat> &y);

void shear_transform(cv::Mat &img, cv::Mat &h0, cv::Mat &h1, vector<cv::Mat> &shear, vector<cv::Mat> &d);

// 2-D FFT 
void fft2(cv::Mat &real, cv::Mat &imag, cv::Mat &fftreal, cv::Mat &fftcomplex, bool shift=false);

void fft2shift(cv::Mat &real, cv::Mat &imag, cv::Mat &fftreal, cv::Mat &fftcomplex);

// 2-D inverse FFT 
void ifft2(cv::Mat &fftreal, cv::Mat &fftcomplex, cv::Mat &real, cv::Mat &imag, bool shift=false);

void ifft2shift(cv::Mat &fftreal, cv::Mat &fftcomplex, cv::Mat &real, cv::Mat &imag) ;



