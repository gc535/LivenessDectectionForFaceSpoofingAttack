#include "math.h" 
#include "LBP.hpp"


//srcImage:灰度图
//LBPImage:LBP图
void LBP::computeLBPImage_256(const cv::Mat &srcImage, cv::Mat &LBPImage)
{
    // 参数检查，内存分配
    CV_Assert(srcImage.depth() == CV_8U&&srcImage.channels() == 1);
    LBPImage.create(srcImage.size(), srcImage.type());

    // 扩充原图像边界，便于边界处理
    cv::Mat extendedImage;
    copyMakeBorder(srcImage, extendedImage, 1, 1, 1, 1, cv::BORDER_DEFAULT);

    // 计算LBP特征图
    int heightOfExtendedImage = extendedImage.rows;
    int widthOfExtendedImage = extendedImage.cols;
    int widthOfLBP=LBPImage.cols;
    uchar *rowOfExtendedImage= extendedImage.data+widthOfExtendedImage+1;
    uchar *rowOfLBPImage = LBPImage.data;
    for (int y = 1; y <= heightOfExtendedImage - 2; ++y, rowOfExtendedImage += widthOfExtendedImage, rowOfLBPImage += widthOfLBP)
    {
        uchar *colOfExtendedImage = rowOfExtendedImage;
        uchar *colOfLBPImage = rowOfLBPImage;
        for (int x = 1; x <= widthOfExtendedImage - 2; ++x,++colOfExtendedImage,++colOfLBPImage)
        {
            // 计算LBP值
            int LBPValue = 0;
            if (colOfExtendedImage[0 - widthOfExtendedImage - 1] >= colOfExtendedImage[0]) LBPValue |= 1 << 7;                
            if (colOfExtendedImage[0 - widthOfExtendedImage] >= colOfExtendedImage[0])     LBPValue |= 1 << 6; 
            if (colOfExtendedImage[0 - widthOfExtendedImage + 1] >= colOfExtendedImage[0]) LBPValue |= 1 << 5; 
            if (colOfExtendedImage[0 + 1] >= colOfExtendedImage[0])                        LBPValue |= 1 << 4; 
            if (colOfExtendedImage[0 + widthOfExtendedImage + 1] >= colOfExtendedImage[0]) LBPValue |= 1 << 3; 
            if (colOfExtendedImage[0 + widthOfExtendedImage] >= colOfExtendedImage[0])     LBPValue |= 1 << 2; 
            if (colOfExtendedImage[0 + widthOfExtendedImage - 1] >= colOfExtendedImage[0]) LBPValue |= 1 << 1; 
            if (colOfExtendedImage[0 - 1] >= colOfExtendedImage[0])                        LBPValue |= 1; 

            colOfLBPImage[0] = LBPValue;
        }  // col
    }// row
}


// 计算等价模式LBP特征图，为了方便表示特征图，58种等价模式表示为1~58,第59种混合模式表示为0
// 注：你可以将第59类混合模式映射为任意数值，因为要突出等价模式特征，所以非等价模式设置为0比较好
void LBP::computeLBPImage_Uniform(const cv::Mat &srcImage, cv::Mat &LBPImage)
{
    // 参数检查，内存分配
    CV_Assert(srcImage.depth() == CV_8U&&srcImage.channels() == 1);
    LBPImage.create(srcImage.size(), srcImage.type());

    // 计算LBP图
    // 扩充原图像边界，便于边界处理
    cv::Mat extendedImage;
    copyMakeBorder(srcImage, extendedImage, 1, 1, 1, 1, cv::BORDER_DEFAULT);

    // 计算LBP
    int heightOfExtendedImage = extendedImage.rows;
    int widthOfExtendedImage = extendedImage.cols;
    int widthOfLBP=LBPImage.cols;
    uchar *rowOfExtendedImage = extendedImage.data+widthOfExtendedImage+1;
    uchar *rowOfLBPImage = LBPImage.data;
    for (int y = 1; y <= heightOfExtendedImage - 2; ++y,rowOfExtendedImage += widthOfExtendedImage, rowOfLBPImage += widthOfLBP)
    {
        // 列
        uchar *colOfExtendedImage = rowOfExtendedImage;
        uchar *colOfLBPImage = rowOfLBPImage;
        for (int x = 1; x <= widthOfExtendedImage - 2; ++x, ++colOfExtendedImage, ++colOfLBPImage)
        {
            // 计算LBP值
            int LBPValue = 0;
            if (colOfExtendedImage[0 - widthOfExtendedImage - 1] >= colOfExtendedImage[0]) LBPValue |= 1 << 7;                
            if (colOfExtendedImage[0 - widthOfExtendedImage] >= colOfExtendedImage[0])     LBPValue |= 1 << 6; 
            if (colOfExtendedImage[0 - widthOfExtendedImage + 1] >= colOfExtendedImage[0]) LBPValue |= 1 << 5; 
            if (colOfExtendedImage[0 + 1] >= colOfExtendedImage[0])                        LBPValue |= 1 << 4; 
            if (colOfExtendedImage[0 + widthOfExtendedImage + 1] >= colOfExtendedImage[0]) LBPValue |= 1 << 3; 
            if (colOfExtendedImage[0 + widthOfExtendedImage] >= colOfExtendedImage[0])     LBPValue |= 1 << 2; 
            if (colOfExtendedImage[0 + widthOfExtendedImage - 1] >= colOfExtendedImage[0]) LBPValue |= 1 << 1; 
            if (colOfExtendedImage[0 - 1] >= colOfExtendedImage[0])                        LBPValue |= 1; 

            // 计算58种等价模式(Uniform Pattern) LBP
            colOfLBPImage[0] = _uniform[_minRI[LBPValue]];
        } // x

    }// y
}

void LBP::computeLBPImage_RI_Uniform(const cv::Mat &srcImage, cv::Mat &LBPImage)
{
    // 参数检查，内存分配
    CV_Assert(srcImage.depth() == CV_8U && srcImage.channels() == 1);
    LBPImage.create(srcImage.size(), srcImage.type());

    // 扩充图像，处理边界情况
    cv::Mat extendedImage;
    copyMakeBorder(srcImage, extendedImage, 1, 1, 1, 1, cv::BORDER_DEFAULT);

    int heigthOfExtendedImage = extendedImage.rows;
    int widthOfExtendedImage = extendedImage.cols;
    int widthOfLBPImage = LBPImage.cols;

    uchar *rowOfExtendedImage = extendedImage.data + widthOfExtendedImage + 1;
    uchar *rowOfLBPImage = LBPImage.data;
    for (int y = 1; y <= heigthOfExtendedImage - 2; ++y, rowOfExtendedImage += widthOfExtendedImage, rowOfLBPImage += widthOfLBPImage)
    {
        // 列
        uchar *colOfExtendedImage = rowOfExtendedImage;
        uchar *colOfLBPImage = rowOfLBPImage;
        for (int x = 1; x <= widthOfExtendedImage - 2; ++x, ++colOfExtendedImage, ++colOfLBPImage)
        {
            // 计算LBP值
            int LBPValue = 0;
            if (colOfExtendedImage[0 - widthOfExtendedImage - 1] >= colOfExtendedImage[0]) LBPValue |= 1 << 7;                
            if (colOfExtendedImage[0 - widthOfExtendedImage] >= colOfExtendedImage[0])     LBPValue |= 1 << 6; 
            if (colOfExtendedImage[0 - widthOfExtendedImage + 1] >= colOfExtendedImage[0]) LBPValue |= 1 << 5; 
            if (colOfExtendedImage[0 + 1] >= colOfExtendedImage[0])                        LBPValue |= 1 << 4; 
            if (colOfExtendedImage[0 + widthOfExtendedImage + 1] >= colOfExtendedImage[0]) LBPValue |= 1 << 3; 
            if (colOfExtendedImage[0 + widthOfExtendedImage] >= colOfExtendedImage[0])     LBPValue |= 1 << 2; 
            if (colOfExtendedImage[0 + widthOfExtendedImage - 1] >= colOfExtendedImage[0]) LBPValue |= 1 << 1; 
            if (colOfExtendedImage[0 - 1] >= colOfExtendedImage[0])                        LBPValue |= 1; 

            // 计算 Rotation Invariant LBP
            int RI = _minRI[LBPValue];
            // 计算58种等价模式(Uniform Pattern) LBP
            int UniformPattern = _uniform[RI];

            // 计算9种等价模式(Rotation Invarition Uniform Pattern) LBP
            colOfLBPImage[0] = getRIUniformPattern(UniformPattern);
        }
    }
}

void LBP::computeLBPFeatureVector_256(const cv::Mat &srcImage, cv::Mat &featureVector)
{
    LBP::computeLBPFeatureVector_256(srcImage, featureVector, _cellSize);
}

void LBP::computeLBPFeatureVector_256(const cv::Mat &srcImage, cv::Mat &featureVector, cv::Size cellSize)
{
    // 参数检查，内存分配
    CV_Assert(srcImage.depth() == CV_8U&&srcImage.channels() == 1);

    cv::Mat LBPImage;
    computeLBPImage_256(srcImage,LBPImage);

    // 计算cell个数
    int widthOfCell = cellSize.width;
    int heightOfCell = cellSize.height;
    int numberOfCell_X = srcImage.cols / widthOfCell;// X方向cell的个数
    int numberOfCell_Y = srcImage.rows / heightOfCell;

    // 特征向量的个数
    int numberOfDimension = 256 * numberOfCell_X*numberOfCell_Y;
    featureVector.create(1, numberOfDimension, CV_32FC1);
    featureVector.setTo(cv::Scalar(0));

    // 计算LBP特征向量
    int stepOfCell=srcImage.cols;
    int pixelCount = cellSize.width*cellSize.height;
    float *dataOfFeatureVector=(float *)featureVector.data;

    // cell的特征向量在最终特征向量中的起始位置
    int index = -256;
    for (int y = 0; y <= numberOfCell_Y - 1; ++y)
    {
        for (int x = 0; x <= numberOfCell_X - 1; ++x)
        {
            index+=256;

            // 计算每个cell的LBP直方图
            cv::Mat cell = LBPImage(cv::Rect(x * widthOfCell, y * heightOfCell, widthOfCell, heightOfCell));
            uchar *rowOfCell=cell.data;
            for(int y_Cell=0;y_Cell<=cell.rows-1;++y_Cell,rowOfCell+=stepOfCell)
            {
                uchar *colOfCell=rowOfCell;
                for(int x_Cell=0;x_Cell<=cell.cols-1;++x_Cell,++colOfCell)
                {
                    ++dataOfFeatureVector[index + colOfCell[0]];
                }
            }

            // normalize
            for (int i = 0; i <= 255; ++i)
                dataOfFeatureVector[index + i] /= pixelCount;
        }
    }

}

// cellSize:每个cell的大小,如16*16
void LBP::computeLBPFeatureVector_Uniform(const cv::Mat &srcImage, cv::Mat &featureVector)
{
    computeLBPFeatureVector_Uniform(srcImage, featureVector, _cellSize);
}

void LBP::computeLBPFeatureVector_Uniform(const cv::Mat &srcImage, cv::Mat &featureVector, cv::Size cellSize)
{
    // 参数检查，内存分配
    CV_Assert(srcImage.depth() == CV_8U&&srcImage.channels() == 1);

    cv::Mat LBPImage;
    computeLBPImage_Uniform(srcImage,LBPImage);

    // 计算cell个数
    int widthOfCell = cellSize.width;
    int heightOfCell = cellSize.height;
    int numberOfCell_X = srcImage.cols / widthOfCell;// X方向cell的个数
    int numberOfCell_Y = srcImage.rows / heightOfCell;

    // 特征向量的个数
    int numberOfDimension = 58 * numberOfCell_X*numberOfCell_Y;
    featureVector.create(1, numberOfDimension, CV_32FC1);
    featureVector.setTo(cv::Scalar(0));

    // 计算LBP特征向量
    int stepOfCell=srcImage.cols;
    int index = -58;// cell的特征向量在最终特征向量中的起始位置
    float *dataOfFeatureVector=(float *)featureVector.data;
    for (int y = 0; y <= numberOfCell_Y - 1; ++y)
    {
        for (int x = 0; x <= numberOfCell_X - 1; ++x)
        {
            index+=58;

            // 计算每个cell的LBP直方图
            cv::Mat cell = LBPImage(cv::Rect(x * widthOfCell, y * heightOfCell, widthOfCell, heightOfCell));
            uchar *rowOfCell=cell.data;
            int sum = 0; // 每个cell的等价模式总数
            for(int y_Cell=0;y_Cell<=cell.rows-1;++y_Cell,rowOfCell+=stepOfCell)
            {
                uchar *colOfCell=rowOfCell;
                for(int x_Cell=0;x_Cell<=cell.cols-1;++x_Cell,++colOfCell)
                {
                    if(colOfCell[0]!=0)
                    {
                        // 在直方图中转化为0~57，所以是colOfCell[0] - 1
                        ++dataOfFeatureVector[index + colOfCell[0]-1];
                        ++sum;
                    }
                }
            }

            // normalize
            for (int i = 0; i <= 57; ++i)
                dataOfFeatureVector[index + i] /= sum;
        }
    }
}

// cellSize:每个cell的大小,如16*16
void LBP::computeLBPFeatureVector_RI_Uniform(const cv::Mat &srcImage, cv::Mat &featureVector)
{
    computeLBPFeatureVector_RI_Uniform(srcImage, featureVector, _cellSize);
}
void LBP::computeLBPFeatureVector_RI_Uniform(const cv::Mat &srcImage, cv::Mat &featureVector, cv::Size cellSize)
{
    // 参数检查，内存分配
    CV_Assert(srcImage.depth() == CV_8U&&srcImage.channels() == 1);

    cv::Mat LBPImage;
    computeLBPImage_RI_Uniform(srcImage,LBPImage);

    // 计算cell个数
    int widthOfCell = cellSize.width;
    int heightOfCell = cellSize.height;
    int numberOfCell_X = srcImage.cols / widthOfCell;// X方向cell的个数
    int numberOfCell_Y = srcImage.rows / heightOfCell;

    // 特征向量的个数
    int numberOfDimension = 9 * numberOfCell_X*numberOfCell_Y;
    featureVector.create(1, numberOfDimension, CV_32FC1);
    featureVector.setTo(cv::Scalar(0));

    // 计算LBP特征向量
    int stepOfCell=srcImage.cols;
    int index = -9;// cell的特征向量在最终特征向量中的起始位置
    float *dataOfFeatureVector=(float *)featureVector.data;
    for (int y = 0; y <= numberOfCell_Y - 1; ++y)
    {
        for (int x = 0; x <= numberOfCell_X - 1; ++x)
        {
            index+=9;
            // 计算每个cell的LBP直方图
            cv::Mat cell = LBPImage(cv::Rect(x * widthOfCell, y * heightOfCell, widthOfCell, heightOfCell));
            uchar *rowOfCell=cell.data;
            int sum = 0; // 每个cell的等价模式总数
            for(int y_Cell=0;y_Cell<=cell.rows-1;++y_Cell,rowOfCell+=stepOfCell)
            {
                uchar *colOfCell=rowOfCell;
                for(int x_Cell=0;x_Cell<=cell.cols-1;++x_Cell,++colOfCell)
                {
                    if(colOfCell[0]!=0)
                    {
                        // 在直方图中转化为0~8，所以是colOfCell[0] - 1
                        ++dataOfFeatureVector[index + colOfCell[0]-1];
                        ++sum;
                    }
                }
            }

            // normalize
            for (int i = 0; i <= 8; ++i)
                dataOfFeatureVector[index + i] /= sum;
        }
    }
}


cv::Mat LBP::mergeRows(const cv::Mat& A, const cv::Mat& B)
{
     int totalRows = A.rows + B.rows;
     cv::Mat mergedDescriptors(totalRows, A.cols, A.type());
     cv::Mat submat = mergedDescriptors.rowRange(0, A.rows);
     A.copyTo(submat);
     submat = mergedDescriptors.rowRange(A.rows, totalRows);
     B.copyTo(submat);
     return mergedDescriptors;
}


//获取i中0,1的跳变次数
int LBP::getHopCount(int i)
{
    // 转换为二进制
    int a[8] = { 0 };
    int k = 7;
    while (i)
    {
        // 除2取余
        a[k] = i % 2;
        i/=2;
        --k;
    }

    // 计算跳变次数
    int count = 0;
    for (int k = 0; k<8; ++k)
    {
        // 注意，是循环二进制,所以需要判断是否为8
        if (a[k] != a[k + 1 == 8 ? 0 : k + 1])
        {
            ++count;
        }
    }
    return count;

}

// 建立等价模式表
// 这里为了便于建立LBP特征图，58种等价模式序号从1开始:1~58,第59类混合模式映射为0
void LBP::buildUniformPatternTable(int *table)
{
    memset(table, 0, 256*sizeof(int));
    uchar temp = 1;
    for (int i = 0; i<256; ++i)
    {
        if (getHopCount(i) <= 2)
        {
            table[i] = temp;
            temp++;
        }
    }
}

float cosine_distance(cv::Mat &A, cv::Mat &B)
{
    float sumxx=0, sumyy=0, sumxy=0; 
    
    for(int r=0; r<A.rows; r++) {
        for(int c=0; c<B.cols; c++) {
              float x = A.at<float>(r,c);
              float y = B.at<float>(r,c);
              sumxx += x*x;
              sumyy += y*y;
              sumxy += x*y;
        } 
    }
    return sumxy/sqrt(sumxx*sumyy); 
}


cv::Mat sobel_image(cv::Mat &src) 
{
  int scale = 1;
  int delta = 0;
  int ddepth = CV_16S;
  cv::Mat grad,src_gray = src;

  cv::GaussianBlur( src, src_gray, cv::Size(3,3), 0, 0, cv::BORDER_DEFAULT );
 
  //cv::cvtColor( src, src_gray, cv::COLOR_BGR2GRAY );
  cv::Mat grad_x, grad_y;
  cv::Mat abs_grad_x, abs_grad_y;

  //Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
  cv::Sobel( src_gray, grad_x, ddepth, 1, 0, 3, scale, delta, cv::BORDER_DEFAULT );
  //Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
  cv::Sobel( src_gray, grad_y, ddepth, 0, 1, 3, scale, delta, cv::BORDER_DEFAULT );
  cv::convertScaleAbs( grad_x, abs_grad_x );
  cv::convertScaleAbs( grad_y, abs_grad_y );
  cv::addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );


  return grad; 
}



