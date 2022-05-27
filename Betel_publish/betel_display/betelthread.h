#ifndef BETELTHREAD_H
#define BETELTHREAD_H

#include <QObject>
#include<QWidget>
#include<math.h>
#include<QMessageBox>
#include<QElapsedTimer>
#include<QMutex>
#include"GenICam/System.h"
#include"Media/ImageConvert.h"
#include"MessageQue.h"
#include"Media/VideoRender.h"
#include"GenICam/Camera.h"
#include"GenICam/StreamSource.h"
//opencv头文件
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include<opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#undef slots
#include<torch/torch.h>
#include "torch/script.h"
#include"ATen/ATen.h"
#include <memory>
#define slots Q_SLOTS

using namespace std;
using namespace cv;

//大华相机帧类
class CFrameInfo:public Dahua::Memory::CBlock{
public:
    BYTE    *m_pImageBuf;
    int           m_nBufferSize;
    int           m_nWidth;
    int           m_nHeight;
    Dahua::GenICam::EPixelType      m_ePixelType;
    int             m_nPaddingX;
    int             m_nPaddingY;
    uint64_t    m_nTimeStamp;
public:
    CFrameInfo(){
        m_pImageBuf = NULL;
        m_nBufferSize = 0;
        m_nWidth = 0;
        m_nHeight = 0;
        m_ePixelType = Dahua::GenICam::gvspPixelMono8;
        m_nPaddingX = 0;
        m_nPaddingY = 0;
        m_nTimeStamp = 0;
    }
    ~CFrameInfo(){}
};

//识别槟榔线程类
class betelThread:public QObject{
    Q_OBJECT
public:
    explicit betelThread(QObject *parent = 0);
    ~betelThread();

    //取流回调函数
    void FrameCallback(const CFrame& frame);
    //显示线程
    void DisplayThreadProc(Dahua::Infra::CThreadLite& lite);
    //设置相机
    void SetCamera(const QString& strKey,int k);
    //采集图像并存入队列
    void collectFrame();
    //断开相机
    void closeFrame();
    //断开采集
    void pauseFrame();
    //设置曝光时间
    bool setExposureTime(double exposureTime);
    //设置增益
    bool SetAdjustPlus(double gainRaw);
    //设置外部触发
    void setLineTriggerConf();
    //空洞填充
    void fillhole(Mat& img);
    //删除次大区域
    void delete_small(Mat& img, Mat& cut_img);
    //计算旋转角度
    void  cal_angle(const Mat& img, float& angle);
    //旋转阈值图像
    void rotate(Mat img, const float& angle, Mat& rotate_img);
    //获取形状参数
    void get_shape_features(const Mat& img, vector<float>& shape_features, Rect& rect_para);
    //获取纹理参数
    void get_context_features(const Mat& img, vector<float>& context_features);
    //重置框选图片灰度等级
    void set_gray(const Mat& img, Mat& set_img, int min_gray, int max_gray, int gray_level);
    //求GLCM矩阵
   void  get_glcm(const Mat& img, int dx, int dy,  vector<vector<float>>& ret);
   //获取pca和相机配置参数
   void get_parameters(double ex_time, const  vector<double> mean_v, const vector<double> std_v, const vector<vector<double>> pca_v);
   //pca处理
   void pca(const  vector<double>& mean_v, const vector<double>& std_v, const vector<vector<double>>& pca_v,
                             const vector<float>& shape_features, const vector<float>& context_features, vector<float>& features);
   //vector转tensor
   void change_tensor(const vector<float>& features, torch::Tensor& input);

    int camera_flag;
    double exposure_time;
    vector<double> mean_vec;
    vector<double> std_vec;
    vector<vector<double>> pca_vec;

private:
    TMessageQue<CFrameInfo>         m_qDisplayFrameQueue;     //显示队列
    Dahua::GenICam::ICameraPtr          m_pCamera;                          //当前相机
    Dahua::GenICam::IStreamSourcePtr         m_pStreamSource;       //拉流
    Dahua::Infra::CThreadLite           m_thdDisplayThread;             //显示线程类
signals:
    void getFrame(uint8_t* pRgbFrameBuf, int nWidth, int nHeight, uint64_t nPixelFormat);

    void emit_img_label(uint8_t* pRgbFrameBuf, int nWidth, int nHeight, uint64_t nPixelFormat, int label, int camera_flag);
private slots:
    void recognicFrame(uint8_t* pRgbFrameBuf, int nWidth, int nHeight, uint64_t nPixelFormat);
};





#endif // BETELTHREAD_H
