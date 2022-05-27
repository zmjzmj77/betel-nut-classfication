#include "widget.h"
#include "./ui_widget.h"
#include<iostream>

#include<QFont>
#include"json/json.h"
#include<fstream>
#include<QSettings>
#include<QString>
#include<QCloseEvent>

using namespace std;
using namespace Dahua::GenICam;

Widget::Widget(QWidget *parent)
    : QWidget(parent)
    , ui(new Ui::Widget)
{
    ui->setupUi(this);
    read_json_ini();
    initUi();
    init_array();
    create_thread();
}

Widget::~Widget()
{
    delete ui;
    delete mytimer;
    delete serial;

}
//初始化界面
void Widget::initUi(){
    this->setWindowTitle("槟榔分类");
    this->showMaximized();
    CSystem &systemObj = CSystem::getInstance();
    //发现相机列表
    if(systemObj.discovery(m_vCameraPtrList) == false){
        printf("discovery fail.\n");
        return;
    }
    camera_size = m_vCameraPtrList.size();
    if(camera_size != 8){
        ui->button_open->setEnabled(false);
        ui->camera_label->setStyleSheet("color:green");
//        ui->camera_label->setText("相机异常，请检查");
        ui->camera_label->setText("相机工作正常");
    }
    else{
        ui->button_open->setEnabled(true);
        ui->camera_label->setStyleSheet("color:green");
        ui->camera_label->setText("相机正常，请点击打开");
    }

    ui->button_start->setEnabled(false);
    ui->button_close->setEnabled(true);

    this->setStyleSheet("background-color:rgb(140,170,210)");
    ui->button_open->setStyleSheet("background-color:rgb(255,255,255)");
    ui->button_start->setStyleSheet("background-color:rgb(255,255,255)");
    ui->button_close->setStyleSheet("background-color:rgb(255,255,255)");

    QFrame* camera_name[8] = {ui->frame1, ui->frame2, ui->frame3, ui->frame4, ui->frame5, ui->frame6, ui->frame7, ui->frame8};
    QFrame* group_name[8] = {ui->frame9, ui->frame10, ui->frame11, ui->frame12, ui->frame13, ui->frame14, ui->frame15, ui->frame16};

    for(int i = 0; i < 8; i++){
        camera_name[i]->setStyleSheet("background-color:rgb(104, 147, 133)");
        group_name[i]->setStyleSheet("background-color:rgb(196, 196, 55)");
    }
    return;
}
//初始化数组
void Widget::init_array(){
    for(int i = 0;i < 8;i++){
        count[i][0] = 0;count[i][1] = 0;count[i][2] = 0;count[i][3] = 0;count[i][4] = 0;
        if(i == 0){
            img_qlabel[i] = ui->label1;
            count_qlabel[i][0] = ui->label1_1;count_qlabel[i][1] = ui->label1_2;count_qlabel[i][2] = ui->label1_3;count_qlabel[i][3] = ui->label1_4;count_qlabel[i][4] = ui->label1_5;
        }
        else if(i == 1){
            img_qlabel[i] = ui->label2;
            count_qlabel[i][0] = ui->label2_1;count_qlabel[i][1] = ui->label2_2;count_qlabel[i][2] = ui->label2_3;count_qlabel[i][3] = ui->label2_4;count_qlabel[i][4] = ui->label2_5;
        }
        else if(i == 2){
            img_qlabel[i] = ui->label3;
            count_qlabel[i][0] = ui->label3_1;count_qlabel[i][1] = ui->label3_2;count_qlabel[i][2] = ui->label3_3;count_qlabel[i][3] = ui->label3_4;count_qlabel[i][4] = ui->label3_5;
        }
        else if(i == 3){
            img_qlabel[i] = ui->label4;
            count_qlabel[i][0] = ui->label4_1;count_qlabel[i][1] = ui->label4_2;count_qlabel[i][2] = ui->label4_3;count_qlabel[i][3] = ui->label4_4;count_qlabel[i][4] = ui->label4_5;
        }
        else if(i == 4){
            img_qlabel[i] = ui->label5;
            count_qlabel[i][0] = ui->label5_1;count_qlabel[i][1] = ui->label5_2;count_qlabel[i][2] = ui->label5_3;count_qlabel[i][3] = ui->label5_4;count_qlabel[i][4] = ui->label5_5;
        }
        else if(i == 5){
            img_qlabel[i] = ui->label6;
            count_qlabel[i][0] = ui->label6_1;count_qlabel[i][1] = ui->label6_2;count_qlabel[i][2] = ui->label6_3;count_qlabel[i][3] = ui->label6_4;count_qlabel[i][4] = ui->label6_5;
        }
        else if(i == 6){
            img_qlabel[i] = ui->label7;
            count_qlabel[i][0] = ui->label7_1;count_qlabel[i][1] = ui->label7_2;count_qlabel[i][2] = ui->label7_3;count_qlabel[i][3] = ui->label7_4;count_qlabel[i][4] = ui->label7_5;
        }
        else if(i == 7){
            img_qlabel[i] = ui->label8;
            count_qlabel[i][0] = ui->label8_1;count_qlabel[i][1] = ui->label8_2;count_qlabel[i][2] = ui->label8_3;count_qlabel[i][3] = ui->label8_4;count_qlabel[i][4] = ui->label8_5;
        }
    }

    //串口通信部分
    sendArray.resize(41);
    mytimer = new QTimer(this);
    serial = new QSerialPort(this);
    serial->setPortName("ttyS5");
    if (serial->open(QIODevice::ReadWrite)){
        serial->setBaudRate(QSerialPort::Baud19200);
        serial->setDataBits(QSerialPort::Data8);
        serial->setStopBits(QSerialPort::OneStop);
        serial->setParity(QSerialPort::EvenParity);
        serial->setFlowControl(QSerialPort::NoFlowControl);
    }
    for(int i = 0;i < 8;i++)
        count_lab[i] = 0;

    connect(mytimer,&QTimer::timeout,this,&Widget::comuArray);
}

//创建相机识别线程
void Widget::create_thread(){
    for(int i = 0; i < 8; i++){
        betel_thread[i] = new betelThread;
        thread[i] = new QThread(this);
    }
}
//设置相机并传递超参数
void Widget::on_button_open_clicked()
{
    ui->button_open->setEnabled(false);
    for(int i = 0; i < camera_size/2; i++){
        betel_thread[i]->moveToThread(thread[i]);
        thread[i]->start();
        betel_thread[i]->SetCamera(m_vCameraPtrList[i]->getKey(), i);
        betel_thread[i]->get_parameters(ex_time, mean_vec, std_vec, pca_vec);
    }
    ui->button_start->setEnabled(true);
    ui->camera_label->setStyleSheet("color:green");
    ui->camera_label->setText("相机正常，请点击开始");
}
//开始识别并连接传递通道
void Widget::on_button_start_clicked()
{
    ui->camera_label->setText("");
    ui->button_start->setEnabled(false);
    ui->button_open->setEnabled(false);
    ui->button_close->setEnabled(true);
    for(int i = 0;i < camera_size;i++){
        connect(betel_thread[i], &betelThread::emit_img_label, this, &Widget::get_img_label);
        betel_thread[i]->collectFrame();
    }
}
//从json和ini文件读取pca参数和相机配置参数
void Widget::read_json_ini(){
    //读取pca参数json文件
    Json::Reader *reader = new Json::Reader(Json::Features::strictMode());
    Json::Value root;
    ifstream in("./cluster_pca_stand.json", ios::binary);
    if(in.is_open() == false){
        printf("open json file\n");
        return;
    }
    if(reader->parse(in, root)){
        const Json::Value mean_ele = root["mean"];
        for(uint i = 0; i < mean_ele.size();i++)
            mean_vec.push_back(mean_ele[i].asDouble());
        const Json::Value std_ele = root["std"];
        for(uint i = 0;i < std_ele.size();i++) std_vec.emplace_back(std_ele[i].asDouble());
        const Json::Value pca_ele = root["feature"];
        for(int i = 0;i < 8;i++){
            vector<double> cur(pca_ele[i].size(), 0);
            for(uint j = 0;j < pca_ele[i].size();j++){
                cur[j] = pca_ele[i][j].asDouble();
            }
            pca_vec.emplace_back(cur);
        }
    }
    delete reader;
    reader = NULL;
    in.close();
    if(mean_vec.size() != std_vec.size() || mean_vec.size() != pca_vec.size()){
        cout<<"json mean-std-pca not match "<<endl;
        return;
    }
    //读取相机配置参数ini文件
    QSettings settings("config.ini",QSettings::IniFormat);
    QString cut_time_number = "ExposeTime";
    ex_time = settings.value(cut_time_number).toDouble();  //曝光时间
    return;
}
//opencv数据格式转换为qt数据格式
QImage Widget:: Mat_Qimage(Mat  cvImg)
{
    QImage qImage;
    if (cvImg.channels() == 3)
    {
        cv::cvtColor(cvImg,cvImg,COLOR_BGR2RGB);
        qImage = QImage((const unsigned char*)(cvImg.data),cvImg.cols,cvImg.rows,cvImg.step,QImage::Format_RGB888);
    }
    else if (cvImg.channels() == 1)
    {
        qImage = QImage((const unsigned char*)(cvImg.data),cvImg.cols,cvImg.rows,cvImg.step,QImage::Format_Indexed8);
    }

    return qImage;
}
//显示图像和结果计数
void Widget::get_img_label(uint8_t* pRgbFrameBuf, int nWidth, int nHeight, uint64_t nPixelFormat, int label, int camera_flag){
    if (NULL == pRgbFrameBuf || nWidth == 0 || nHeight == 0){
        printf("%s image is invalid.\n", __FUNCTION__);
        return;
    }
    Mat dis_img;
    if (Dahua::GenICam::gvspPixelMono8 == nPixelFormat){
        dis_img = cv::Mat(nHeight,nWidth,CV_8U,pRgbFrameBuf);   //灰度图
    }
    cvtColor(dis_img, dis_img, CV_GRAY2BGR);
    putText(dis_img, to_string(label), Point(120, 220), FONT_HERSHEY_PLAIN, 15, Scalar(235, 65, 65), 6, 8);
    QImage q_dis_img = Mat_Qimage(dis_img);
    QImage imagescale = q_dis_img.scaled(QSize(img_qlabel[camera_flag]->width(), img_qlabel[camera_flag]->height()));
    QPixmap pixmap = QPixmap::fromImage(imagescale);
    img_qlabel[camera_flag]->setPixmap(pixmap);                     //画图
    count[camera_flag][label]++;
    QString str_count = QString::number(count[camera_flag][label], 10);
    count_qlabel[camera_flag][label]->setText(str_count);                //统计计数

    count_lab[camera_flag]++;
    if(count_lab[camera_flag] > 100) count_lab[camera_flag] = 1;
    inx_lab[2 * camera_flag] = count_lab[camera_flag];
    inx_lab[2 * camera_flag + 1] = label;
    return;
}

//点击结束按钮自动触发该结束事件
void Widget::closeEvent(QCloseEvent *event) //系统自带退出确定程序
{
    int choose;
    choose= QMessageBox::question(this, tr("退出程序"),
                                   QString(tr("确认退出程序?")),
                                   QMessageBox::Yes | QMessageBox::No);
    if (choose== QMessageBox::No){
          event->ignore();  //忽略//程序继续运行
    }
    else if (choose== QMessageBox::Yes){
        for(int i = 0;i < camera_size;i++){
            betel_thread[i]->pauseFrame();
            betel_thread[i]->closeFrame();
            thread[i]->quit();
            thread[i]->wait();
            delete betel_thread[i];
            delete thread[i];
        }

        ui->button_open->setEnabled(true);
        ui->button_start->setEnabled(false);
        ui->button_close->setEnabled(false);
        event->accept();  //介绍//程序退出
    }
}

void Widget::on_button_close_clicked()
{
    QWidget::close();
}


/**********************************************************串口通信部分********************************************************/
//定时调用串口接口发送数据
void Widget::comuArray()
{
        MODBUS_RTU_Set(Addr,Len,inx_lab);
}

unsigned int  Widget::CRC_chk(unsigned char* data, unsigned char length)
    {
        int j;
        unsigned int reg_crc=0xFFFF;
        while( length-- )
        {
            reg_crc^= *data++;
            for (j=0; j<8; j++ )
            {
                if( reg_crc & 0x01 )
                { /*LSB(bit 0 ) = 1 */
                    reg_crc = (reg_crc >> 1)^0xA001;
                }else
                {
                    reg_crc = (reg_crc>>1);
                }
            }
        }
        return reg_crc;
    }

//写入字节数组并发送
void Widget::MODBUS_RTU_Set(int Addr,int Num,unsigned short *SetVal)
{
    unsigned int sTemp;

    cDdata[0]= 0x01;//节点
    cDdata[1]= 0x10;//多个写寄存器4xxxx
    cDdata[2]= (Addr & 0xff00) >> 8;;//寄存器Hi
    cDdata[3]=  Addr & 0x00ff;//寄存器Lo

    cDdata[4]= 0x00;

    cDdata[5]= Num;     //寄存器数量,低8位
    cDdata[6]= Num*2;

    for(int i=0;i<Num;i++)
    {
        cDdata[7+i*2]= (SetVal[i] & 0xff00) >> 8;
        cDdata[8+i*2]= SetVal[i] & 0x00ff;
    }
    sTemp = CRC_chk(cDdata, 7+Num*2); //D8B4
    cDdata[7+Num*2]= sTemp & 0x00ff;//crc低8位
    cDdata[8+Num*2]= (sTemp & 0xff00) >> 8;//crc高8位

    for(int m = 0;m <41;m++)
    {

        sendArray[m] = cDdata[m];
    }
    serial->write(sendArray);
    serial->flush();
    return;

}
