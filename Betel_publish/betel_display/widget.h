#ifndef WIDGET_H
#define WIDGET_H

#include<iostream>
#include <QWidget>
#include<QThread>
#include<QTimer>
#include<QtSerialPort/QSerialPort>
#include <QtSerialPort/QSerialPortInfo>

#include"GenICam/System.h"

#include"betelthread.h"

QT_BEGIN_NAMESPACE
namespace Ui { class Widget; }
QT_END_NAMESPACE

class Widget : public QWidget
{
    Q_OBJECT

public:
    Widget(QWidget *parent = nullptr);
    ~Widget();

    void  read_json_ini();

    QImage Mat_Qimage(Mat  cvImg);

    betelThread* betel_thread[8];
    QThread* thread[8];

    vector<double> mean_vec;
    vector<double> std_vec;
    vector<vector<double>> pca_vec;
    double ex_time;

    //串口通信部分
    QByteArray sendArray;
    QTimer *mytimer;
    QSerialPort *serial;
    unsigned short inx_lab[16];  //正常发送
    unsigned short count_lab[8];
    unsigned char cDdata[64];
    int Addr = 0X1000;//D0地址
    int Len = 16;//D寄存器数量
    unsigned int  CRC_chk(unsigned char* data, unsigned char length);
    void MODBUS_RTU_Set(int Addr,int Num,unsigned short *SetVal);
    bool  worr = true;


signals:
    void a(int* n, int m);

private slots:
    void on_button_open_clicked();
    void on_button_start_clicked();
    //显示图像和结果计数
    void get_img_label(uint8_t* pRgbFrameBuf, int nWidth, int nHeight, uint64_t nPixelFormat, int label, int camera_flag);

    void on_button_close_clicked();

    void comuArray();
private:
    Ui::Widget *ui;

    //变量
    Dahua::Infra::TVector<Dahua::GenICam::ICameraPtr> m_vCameraPtrList;	// 发现的相机列表

    int camera_size;

    QLabel* img_qlabel[8];
    QLabel* count_qlabel[8][5];
    int count[8][5];


    //函数
    void  initUi();
    void  create_thread();
    void init_array();
    void closeEvent(QCloseEvent *event);

};
#endif // WIDGET_H
