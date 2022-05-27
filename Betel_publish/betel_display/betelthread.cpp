#include"betelthread.h"
#include<iostream>
#include<QThread>
#include<math.h>

betelThread::betelThread(QObject *parent):
    QObject(parent),
    m_thdDisplayThread(Dahua::Infra::CThreadLite::ThreadProc(&betelThread::DisplayThreadProc, this), "Display")
{
    qRegisterMetaType<uint64_t>("uint64_t");
    qRegisterMetaType<string>("string");
//启动显示线程
    if(m_thdDisplayThread.isThreadOver() == false)
    {
          m_thdDisplayThread.destroyThread();
     }
     m_thdDisplayThread.createThread();

     connect(this, &betelThread::getFrame, this, &betelThread::recognicFrame);
}

betelThread::~betelThread(){

}

//显示线程
void betelThread::DisplayThreadProc(Dahua::Infra::CThreadLite& lite)
{
    while (lite.looping())
    {
        CFrameInfo frameinfo;

        if (false == m_qDisplayFrameQueue.get(frameinfo, 500))
        {
            continue;
        }
        emit getFrame(frameinfo.m_pImageBuf, frameinfo.m_nWidth, frameinfo.m_nHeight, frameinfo.m_ePixelType);
       free(frameinfo.m_pImageBuf);
    }
}

// 取流回调函数（将采集的图像先放进缓冲区，缓冲区最多存16张）
void betelThread::FrameCallback(const CFrame& frame)
{
    CFrameInfo frameInfo;
    frameInfo.m_nWidth = frame.getImageWidth();
    frameInfo.m_nHeight = frame.getImageHeight();
   frameInfo.m_nBufferSize = frame.getImageSize();
    frameInfo.m_nBufferSize =  frame.getImageWidth()*frame.getImageHeight()*3;
    frameInfo.m_nPaddingX = frame.getImagePadddingX();
    frameInfo.m_nPaddingY = frame.getImagePadddingY();
    frameInfo.m_ePixelType = frame.getImagePixelFormat();
    frameInfo.m_pImageBuf = (BYTE *)malloc(sizeof(BYTE)* frameInfo.m_nBufferSize);   //申请sizeof(BYTE)* n的内存空间n×sizeof(BYTE*)
    frameInfo.m_nTimeStamp = frame.getImageTimeStamp();

    /* 内存申请失败，直接返回 */
    if (frameInfo.m_pImageBuf != NULL)
    {
        memcpy(frameInfo.m_pImageBuf, frame.getImage(), frame.getImageSize());

        if (m_qDisplayFrameQueue.size() > 16)
        {
            CFrameInfo frameOld;
            m_qDisplayFrameQueue.get(frameOld);                                         //得到当前队列的第一个元素值
            free(frameOld.m_pImageBuf);                                                 //释放掉第一个元素的内存
        }
        m_qDisplayFrameQueue.push_back(frameInfo);
    }
}

// 设置当前相机
void betelThread::SetCamera(const QString& strKey,int k)
{
    CSystem &systemObj = CSystem::getInstance();
    m_pCamera = systemObj.getCameraPtr(strKey.toStdString().c_str());
    camera_flag = k;
}

//设置曝光时间
bool betelThread::setExposureTime(double exposureTime)
{

    if (NULL == m_pCamera)
    {
        printf("Set ExposureTime fail. No camera or camera is not connected.\n");
        return false;
    }

    CDoubleNode nodeExposureTime(m_pCamera, "ExposureTime");

    if (false == nodeExposureTime.isValid())
    {
        printf("get ExposureTime node fail.\n");
        return false;
    }

    if (false == nodeExposureTime.isAvailable())
    {
        printf("ExposureTime is not available.\n");
        return false;
    }

    if (false == nodeExposureTime.setValue(exposureTime))
    {
        printf("set ExposureTime value = %f fail.\n", exposureTime);
        return false;
    }

    return true;
}

//设置增益
bool betelThread::SetAdjustPlus(double gainRaw)
{
    if (NULL == m_pCamera)
    {
        printf("Set GainRaw fail. No camera or camera is not connected.\n");
        return false;
    }

    CDoubleNode nodeGainRaw(m_pCamera, "GainRaw");

    if (false == nodeGainRaw.isValid())
    {
        printf("get GainRaw node fail.\n");
        return false;
    }

    if (false == nodeGainRaw.isAvailable())
    {
        printf("GainRaw is not available.\n");
        return false;
    }

    if (false == nodeGainRaw.setValue(gainRaw))
    {
        printf("set GainRaw value = %f fail.\n", gainRaw);
        return false;
    }

    return true;
 }

 //设置外部触发
void betelThread::setLineTriggerConf()
{
    //设置触发源为Line1触发
    CEnumNode nodeTriggerSource(m_pCamera, "TriggerSource");
    if (false == nodeTriggerSource.isValid())
    {
        printf("get TriggerSource node fail.\n");
        return;
    }
    if (false == nodeTriggerSource.setValueBySymbol("Line1"))
    {
        printf("set TriggerSource value = Line1 fail.\n");
        return;
    }

    //设置触发器
    CEnumNode nodeTriggerSelector(m_pCamera, "TriggerSelector");
    if (false == nodeTriggerSelector.isValid())
    {
        printf("get TriggerSelector node fail.\n");
        return;
    }
    if (false == nodeTriggerSelector.setValueBySymbol("FrameStart"))
    {
        printf("set TriggerSelector value = FrameStart fail.\n");
        return;
    }

    //设置触发模式
    CEnumNode nodeTriggerMode(m_pCamera, "TriggerMode");
    if (false == nodeTriggerMode.isValid())
    {
        printf("get TriggerMode node fail.\n");
        return;
    }
    if (false == nodeTriggerMode.setValueBySymbol("On"))
    {
        printf("set TriggerMode value = On fail.\n");
        return;
    }

    // 设置外触发为上升沿（下降沿为FallingEdge）
    CEnumNode nodeTriggerActivation(m_pCamera, "TriggerActivation");
    if (false == nodeTriggerActivation.isValid())
    {
        printf("get TriggerActivation node fail.\n");
        return;
    }
    if (false == nodeTriggerActivation.setValueBySymbol("RisingEdge"))
    {
        printf("set TriggerActivation value = RisingEdge fail.\n");
        return;
    }
    return;
}

//采集帧
void betelThread::collectFrame()
{
    //判断相机是否正确连接
    if (NULL == m_pCamera)
    {
        printf("connect camera fail. No camera.\n");
        return;
    }
    if (true == m_pCamera->isConnected())
    {
        printf("camera is already connected.\n");
        return;
    }

    if (false == m_pCamera->connect())
    {
        printf("connect camera fail.\n");
        return;
    }

    //设置曝光时间、增益、外触发
    setExposureTime(exposure_time);
    //SetAdjustPlus(100.0);
   setLineTriggerConf();

    //创建流对象
       if (NULL == m_pStreamSource)
       {
           m_pStreamSource = CSystem::getInstance().createStreamSource(m_pCamera);     //创建流对象
       }
       if (NULL == m_pStreamSource)
       {
           return ;
       }

       if (m_pStreamSource->isGrabbing())
       {
           return ;
       }

       //以注册回调的方式获取图像，存进缓冲区
       bool bRet = m_pStreamSource->attachGrabbing(IStreamSource::Proc(&betelThread::FrameCallback, this));   //注册回调函数，为类成员函数
       if (!bRet)
       {
           return ;
       }

       if (!m_pStreamSource->startGrabbing())
       {
           return ;
       }

       return;
}

//获取pca参数和相机配置参数
void betelThread::get_parameters(double ex_time, const vector<double> mean_v, const vector<double> std_v, const vector<vector<double> > pca_v){
    exposure_time = ex_time;
    mean_vec = mean_v;
    std_vec = std_v;
    pca_vec = pca_v;
}

//空洞填充
void betelThread::fillhole(Mat& img){
    Size img_size = img.size();
    Mat mask = Mat::zeros(img_size.height+2, img_size.width+2, img.type());
    bool isbreak = false;
    Point seedpoint;
    for(int i = 0;i < img_size.height;i++){
        for(int j = 0;j < img_size.width;j++){
            if(img.ptr<uchar>(i)[j] == 0){
                seedpoint = Point(i, j);
                isbreak = true;
                break;
            }
        }
        if(isbreak == true) break;
    }
    Mat img_c = img.clone();
    floodFill(img_c, mask, seedpoint, 255);
    img = img | (~img_c);
    return;
}
//删除次大区域
void betelThread::delete_small(Mat &img, Mat &cut_img){
    vector<vector<Point>> contours;
    findContours(img, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
    int index= 0;
    double area = 0;
    for(uint i = 0;i < contours.size();i++){
        double cur_area = contourArea(contours[i]);
        if (cur_area > area){
            index = i;
        }
    }
    for(uint i = 0;i < contours.size();i++){
        if(i == index) continue;
        drawContours(img, contours[i], -1, 0, -1);
    }
    Rect coor_tuple = boundingRect(contours[index]); //(tl.x, tl.y,w,h)
    cut_img = img(coor_tuple);
    return;
}
//计算旋转角度
void betelThread::cal_angle(const Mat& img, float &angle){
    vector<Point> middle_coor;
    for(int i = 0;i < img.rows;i++){
        vector<int> col_coor;
        for(int j = 0;j < img.cols;j++){
            if(img.ptr<uchar>(i)[j] != 0 )
                col_coor.emplace_back(j);
        }
        if(col_coor.size() == 0) continue;
        int col_mid = col_coor[int(col_coor.size()/2)];
        middle_coor.push_back(Point(i, col_mid));
    }
    Vec4f line;  //(vx, vx, x0, y0) 直线在x,y方向的方向向量和直线上的一个点
    fitLine(middle_coor, line, CV_DIST_L2, 0, 0.01, 0.01);
    angle = atan(line[1] / line[0]);
    if(angle < 0) angle = M_PI / 2 - abs(angle);
    else angle = -(M_PI / 2 - angle);
    return;
}
//旋转阈值图像
void betelThread::rotate(Mat img, const float &angle, Mat& rotate_img){
    Size img_size = img.size();
    Point2f center_coor = Point(img_size.height / 2, img_size.width / 2);
    Mat M = getRotationMatrix2D(center_coor, angle / M_PI * 180, 1.0);
    warpAffine(img, rotate_img, M, img_size);
    return;
}
//获取形状参数
void betelThread::get_shape_features(const Mat& img, vector<float>& shape_features, Rect& rect_para){
    vector<vector<Point>> contours;
    findContours(img, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
    int index= 0;
    float area = 0;
    for(int i = 0;i < contours.size();i++){
        double cur_area = contourArea(contours[i]);
        if (cur_area > area){
            index = i;
        }
    }
    area = contourArea(contours[index]);  //面积
    rect_para = boundingRect(contours[index]);
    float w = rect_para.width;  //宽
    float h = rect_para.height;  //高
    float scale_hw = h / w;  //高宽比
    float length = arcLength(contours[index], true);  //周长
    float area_scale = area / (w * h);  //轮廓面积/最小外接矩形面积
    vector<Point> hull_points;
    convexHull(contours[index], hull_points);
    float hull_area = contourArea(hull_points);
    float hull_scale = area / hull_area;    //凸性
    float Rc = 4 * M_PI * area / pow(length, 2);   //圆度
    shape_features = {area, w, h, scale_hw, length, area_scale, hull_scale, Rc};
    return;
}
//获取纹理参数
void betelThread::get_context_features(const Mat &img, vector<float> &context_features){
    int gray_level = 8;
    int min_gray = 0, max_gray = 256;
    Mat set_img;
    set_gray(img, set_img, min_gray, max_gray, gray_level);
    vector<vector<float>> ret(gray_level, vector<float>(gray_level, 0));
    get_glcm(set_img, 1, 0, ret);
    float energy = 0, contrast = 0, Idm = 0, entropy = 0;
    for(int i = 0;i < ret.size();i++){
        for(int j = 0;j < ret[0].size();j++){
            energy += pow(ret[i][j], 2);
            contrast += pow(i-j, 2) * ret[i][j];
            Idm += ret[i][j] / (1+pow(i-j, 2));
            if(ret[i][j] > 0) entropy += ret[i][j] * log(ret[i][j]);
        }
    }
    context_features = {energy, contrast, Idm, entropy};
    return;
}
//重置框选图片灰度等级
void betelThread::set_gray(const Mat &img, Mat &set_img, int min_gray, int max_gray, int gray_level){
    set_img = img.clone();
    for(int i = 0;i < img.rows;i++){
        for(int j = 0;j < img.cols;j++){
            set_img.ptr<uchar>(i)[j] = int((img.ptr<uchar>(i)[j] - min_gray) / (max_gray - min_gray) * (gray_level - 1));  //重置像素值
        }
    }
    return;
}
//求GLCM矩阵
void betelThread::get_glcm(const Mat &img, int dx, int dy,  vector<vector<float>>& ret){
    for(int i = 0; i < img.rows-dy;i++){
        for(int j = 0;j < img.cols-dx;j++){
            int left_value = img.ptr<uchar>(i)[j];
            int right_value = img.ptr<uchar>(i+dy)[j+dx];
            ret[left_value][right_value]++;
        }
    }
    for(int i = 0;i < ret.size();i++){
        for(int j = 0;j < ret[0].size();j++){
            ret[i][j] /= float(img.cols * img.rows);
        }
    }
    return;
}
//pca处理
void betelThread::pca(const vector<double> &mean_v, const vector<double> &std_v, const vector<vector<double> > &pca_v,
                      const vector<float> &shape_features, const vector<float> &context_features, vector<float> &features){
    vector<float> fushion_features;
    for(uint i = 0;i < shape_features.size();i++) fushion_features.emplace_back(shape_features[i]);
    for(uint i = 0;i < context_features.size();i++) fushion_features.emplace_back(context_features[i]);
    if(fushion_features.size() != mean_v.size()) {
        cout<<"fushion features not match mean"<<endl;
        return;
    }
    for(uint i = 0;i < fushion_features.size();i++)
        fushion_features[i] = (fushion_features[i] - mean_v[i]) / std_v[i];
    int pca_dim = pca_v[0].size();
    features.resize(pca_dim, 0);
    for(int i = 0;i < pca_dim;i++){
        for(uint j = 0; j < fushion_features.size();j++){
            features[i] += fushion_features[j] * pca_v[j][i];    //矩阵运算
        }
    }
    return;
}
//vector转tensor
void betelThread::change_tensor(const vector<float> &features, torch::Tensor &input){
    int dim = features.size();
    input = torch::zeros({1, dim, 1, 1}, torch::kFloat);
    for(int i = 0;i < dim;i++)
        input[0][i][0][0] = features[i];
    return;
}
//取帧识别并发送结果
void betelThread::recognicFrame(uint8_t* pRgbFrameBuf, int nWidth, int nHeight, uint64_t nPixelFormat){
    if (NULL == pRgbFrameBuf || nWidth == 0 || nHeight == 0){
        printf("%s image is invalid.\n", __FUNCTION__);
        return;
    }

    Mat srcimg, crop_img, gray_img, res_img, thresh_img, rotate_img, rotate_gray_img;
    Mat cut_img, cut_gray_img;
    if (Dahua::GenICam::gvspPixelMono8 == nPixelFormat)
    {
        srcimg = cv::Mat(nHeight,nWidth,CV_8U,pRgbFrameBuf);   //灰度图
    }

    crop_img = srcimg(Rect(srcimg.cols/2-500, srcimg.rows/2-500, srcimg.cols/2+500, srcimg.rows/2+500));
    resize(crop_img, res_img, Size(224, 224));
    threshold(res_img, thresh_img, 180, 255, THRESH_BINARY);
    fillhole(thresh_img);
    delete_small(thresh_img, cut_img);
    float rotate_angle;
    cal_angle(thresh_img, rotate_angle);
    rotate(thresh_img, rotate_angle, rotate_img);
    vector<float> shape_features(8, 0);  //形状特征参数
    Rect rect_para;
    get_shape_features(thresh_img, shape_features, rect_para);
    rotate(res_img, rotate_angle, rotate_gray_img);
    cut_gray_img = rotate_gray_img(Rect(rect_para.x+int(rect_para.width/5), rect_para.y+int(rect_para.height/3), int(rect_para.width*3/5), int(rect_para.height/3)));
    vector<float> context_features(4, 0);
    get_context_features(cut_gray_img, context_features); //纹理特征参数
    vector<float> features;
    pca(mean_vec, std_vec, pca_vec, shape_features, context_features, features);
    torch::Tensor input;
    change_tensor(features, input);
    input = input.to(torch::kCUDA);
    torch::jit::script::Module model = torch::jit::load("./betelnet.pt");
    torch::Tensor output = model.forward({input}).toTensor();
    torch::Tensor output_softmax = torch::softmax(output, 1);
    output_softmax = output_softmax.to(torch::kCPU);

    vector<float> output_vec(output_softmax.data_ptr<float>(),output_softmax.data_ptr<float>()+output_softmax.numel());
    std::vector<float>::iterator biggest = max_element(begin(output_vec),end(output_vec));
    int label = distance(begin(output_vec),biggest);

    emit emit_img_label(pRgbFrameBuf, nWidth, nHeight, nPixelFormat, label, camera_flag);

}

//断开采集
void betelThread::pauseFrame(){
    if (m_pStreamSource != NULL){
        m_pStreamSource->detachGrabbing(IStreamSource::Proc(&betelThread::FrameCallback, this));

        m_pStreamSource->stopGrabbing();
        m_pStreamSource.reset();
    }

    /* 清空显示队列 */
    m_qDisplayFrameQueue.clear();
    return;
}

//断开相机
void betelThread::closeFrame(){
    if (NULL == m_pCamera){
        printf("disconnect camera fail. No camera.\n");
        return ;
    }
    if (false == m_pCamera->isConnected()){
        printf("camera is already disconnected.\n");
        return ;
    }
    if (false == m_pCamera->disConnect()){
        printf("disconnect camera fail.\n");
        return ;
    }
    return;
}
