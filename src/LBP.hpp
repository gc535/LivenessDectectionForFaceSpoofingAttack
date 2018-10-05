
#ifndef LBP_HPP
#define LBP_HPP

#include <iostream>
#include <fstream>
#include "opencv2/opencv.hpp"

using namespace std;

class LBP
{

private: 
    int _rows;
    int _cols;
    cv::Size _cellSize;

    const int _minRI[256] = {
        0, 1, 1, 3, 1, 5, 3, 7, 1, 9, 5, 11, 3, 13, 7, 15, 1, 17, 9, 19, 5, 21, 11, 23, 
        3, 25, 13, 27, 7, 29, 15, 31, 1, 9, 17, 25, 9, 37, 19, 39, 5, 37, 21, 43, 11, 45,
        23, 47, 3, 19, 25, 51, 13, 53, 27, 55, 7, 39, 29, 59, 15, 61, 31, 63, 1, 5, 9, 13, 
        17, 21, 25, 29, 9, 37, 37, 45, 19, 53, 39, 61, 5, 21, 37, 53, 21, 85, 43, 87, 11, 
        43, 45, 91, 23, 87, 47, 95, 3, 11, 19, 27, 25, 43, 51, 59, 13, 45, 53, 91, 27, 91, 
        55, 111, 7, 23, 39, 55, 29, 87, 59, 119, 15, 47, 61, 111, 31, 95, 63, 127, 1, 3, 5, 
        7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 9, 25, 37, 39, 37, 43, 45, 47, 19, 
        51, 53, 55, 39, 59, 61, 63, 5, 13, 21, 29, 37, 45, 53, 61, 21, 53, 85, 87, 43, 91, 
        87, 95, 11, 27, 43, 59, 45, 91, 91, 111, 23, 55, 87, 119, 47, 111, 95, 127, 3, 7, 
        11, 15, 19, 23, 27, 31, 25, 39, 43, 47, 51, 55, 59, 63, 13, 29, 45, 61, 53, 87, 91, 
        95, 27, 59, 91, 111, 55, 119, 111, 127, 7, 15, 23, 31, 39, 47, 55, 63, 29, 61, 87, 
        95, 59, 111, 119, 127, 15, 31, 47, 63, 61, 95, 111, 127, 31, 63, 95, 127, 63, 127, 
        127, 255};

    const int _uniform[256] = { 
        1, 2, 3, 4, 5, 0, 6, 7, 8, 0, 0, 0, 9, 0, 10, 11, 12, 0, 0, 0, 0, 0, 0, 0, 13, 0, 0, 
        0, 14, 0, 15, 16, 17, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18, 0, 0, 0, 0, 0, 
        0, 0, 19, 0, 0, 0, 20, 0, 21, 22, 23, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
        0, 0, 0, 25, 0, 0, 0, 0, 0, 0, 0, 26, 0, 0, 0, 27, 0, 28, 29, 30, 31, 0, 32, 0, 0, 0, 
        33, 0, 0, 0, 0, 0, 0, 0, 34, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 35, 0, 0, 0, 
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0, 36, 
        37, 38, 0, 39, 0, 0, 0, 40, 0, 0, 0, 0, 0, 0, 0, 41, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
        0, 0, 0, 0, 42, 43, 44, 0, 45, 0, 0, 0, 46, 0, 0, 0, 0, 0, 0, 0, 47, 48, 49, 0, 50, 0, 
        0, 0, 51, 52, 53, 0, 54, 55, 56, 57, 58 };

public:

    LBP(cv::Size cellSize=cv::Size(16, 16)): _cellSize(cellSize){}
    LBP(int size_x, int size_y, cv::Size cellSize=cv::Size(16, 16)): _rows(size_y), _cols(size_x), _cellSize(cellSize){}

    /* 计算LBP特征图 */
    void computeLBPImage_256(const cv::Mat &srcImage, cv::Mat &LBPImage);         // 计算256维基本LBP特征图
    void computeLBPImage_Uniform(const cv::Mat &srcImage, cv::Mat &LBPImage);     // 计算等价模式LBP特征图(58种模式)
    void computeLBPImage_RI_Uniform(const cv::Mat &srcImage, cv::Mat &LBPImage);  // 计算灰度不变+旋转不变+等价模式LBP特征图(9种模式)
    
    /* 计算LBP特征向量 */
    void computeLBPFeatureVector_256(const cv::Mat &srcImage, cv::Mat &featureVector);                                  // 计算基本的256维LBP特征向量
    void computeLBPFeatureVector_256(const cv::Mat &srcImage, cv::Mat &featureVector, cv::Size cellSize);     // 计算基本的256维LBP特征向量
    void computeLBPFeatureVector_Uniform(const cv::Mat &srcImage, cv::Mat &featureVector);                              // 计算灰度不变+等价模式LBP特征向量(58种模式)
    void computeLBPFeatureVector_Uniform(const cv::Mat &srcImage, cv::Mat &featureVector, cv::Size cellSize); // 计算灰度不变+等价模式LBP特征向量(58种模式)
    void computeLBPFeatureVector_RI_Uniform(const cv::Mat &input, cv::Mat &featureVector);                              // 计算灰度不变+旋转不变+等价模式LBP特征向量(9种模式)
    void computeLBPFeatureVector_RI_Uniform(const cv::Mat &input, cv::Mat &featureVector, cv::Size cellSize); // 计算灰度不变+旋转不变+等价模式LBP特征向量(9种模式)


private:
    /* find (min) rotation invariant value */
    int getRI(int value256){return _minRI[value256];}   

    /* find 9 rotation invariant uniform pattern */
    int getRIUniformPattern(int value58)   
    {
        int value9 = 0;
        switch (value58)
        {
            case 1:
                value9 = 1;
                break;
            case 2:
                value9 = 2;
                break;
            case 4:
                value9 = 3;
                break;
            case 7:
                value9 = 4;
                break;
            case 11:
                value9 = 5;
                break;
            case 16:
                value9 = 6;
                break;
            case 22:
                value9 = 7;
                break;
            case 29:
                value9 = 8;
                break;
            case 58:
                value9 = 9;
                break;
            default:
                value9 = 0;
                break;
        }
        return value9;
    }

    void buildUniformPatternTable(int *table); // 计算等价模式查找表
    int getHopCount(int i);// 获取i中0,1的跳变次数
};

#endif //LBP_HPP
