#include<opencv2/opencv.hpp>
#include<queue>
#include<algorithm>

using namespace std;
using namespace cv;

// {-5, 1}, {5, -1}, {-4, 1}, {4, -1}, {-10,2}, {10,-2}
static int NeighborDirection[8][2] = {{0,1},{1,1},{1,0},{1,-1},{0,-1},{-1,-1},{-1,0},{-1,1}};

typedef struct Coordinate{
    int x;
    int y;
}Coordinate;

void SearchNeighbor(Mat img_added, Mat &label_mat, queue<Coordinate> &label_coordinate_queue);
void Blur(Mat &img_interpolation, Mat label_mat, int row, int col, String pattern, int ksize);

int main(int argc, char *argv[])
{
    Mat img_src = imread("/Users/keyanchen/Files/Code/LineDetection/LineDetection/test.jpg", IMREAD_GRAYSCALE);
    imwrite("/Users/keyanchen/Files/Code/LineDetection/LineDetection/1.jpg", img_src);
    Mat X_gradient, Y_gradient, gradient_1, gradient_1_cv, gradient_2;
    // X方向一阶梯度
    Sobel(img_src, X_gradient, CV_16S, 1, 0);
    // Y方向一阶梯度
    Sobel(img_src, Y_gradient, CV_16S, 1, 0);
    // 二阶梯度
    Laplacian(img_src, gradient_2, CV_16S, 3);
    // 一阶梯度
    gradient_1 = abs(X_gradient) + abs(Y_gradient);
    // 自带的一阶梯度
    Sobel(img_src, gradient_1_cv, CV_16S, 1, 1);

    // 波谷矩阵
    Mat valley = Mat::zeros(img_src.rows, img_src.cols, CV_8UC1);
    // 统计灰度值对应的数量
    vector<vector<int>> gray_num(256);
    for(int i=0; i<img_src.rows; i++){
        for(int j=0; j<img_src.cols; j++){
            gray_num.at(int(img_src.at<uchar>(i,j))).push_back(1);
            if(gradient_2.at<short>(i, j) > 350){
                valley.at<uchar>(i, j) = 255;
            }
        }
    }

    // 统计阈值
    int threshold_num = int(img_src.rows*img_src.cols*0.02);
    int temp = 0;
    int threshold_gray = 0;
    for(int i=0; i<256; i++){
        if(temp >= threshold_num){
            threshold_gray = i;
            break;
        }
        temp += gray_num.size();
    }

    // 统计阈值化
    Mat img_threshold_1;
    threshold(img_src, img_threshold_1, threshold_gray, 255, THRESH_BINARY_INV);
    // 3*3
    Mat element = getStructuringElement(MORPH_ELLIPSE,Size(3,3));
    morphologyEx(img_threshold_1, img_threshold_1, MORPH_CLOSE, element);

    // 转化为UChar
    gradient_2.convertTo(gradient_2, CV_8UC1);
    valley.convertTo(valley, CV_8UC1);
    imshow("Gradient_2", gradient_2);
    imwrite("/Users/keyanchen/Files/Code/LineDetection/LineDetection/2.jpg", gradient_2);
    imshow("Valley",valley);
    imwrite("/Users/keyanchen/Files/Code/LineDetection/LineDetection/3.jpg", valley);
    imshow("Img_Threshold_1",img_threshold_1);
    imwrite("/Users/keyanchen/Files/Code/LineDetection/LineDetection/4.jpg", img_threshold_1);

    // 波谷检测和统计阈值化的整合
    Mat img_added;
    addWeighted(valley, 0.5, img_threshold_1, 0.5, 0, img_added);
    threshold(img_added, img_added, 126, 255, THRESH_BINARY);

    imshow("Img_Added", img_added);
    imwrite("/Users/keyanchen/Files/Code/LineDetection/LineDetection/5.jpg", img_added);

    // 统计霍夫变换
    Mat img_src_color;
    vector<cv::Vec2f> lines;
    cvtColor(img_src, img_src_color, COLOR_GRAY2BGR);
    HoughLines(img_added, lines, 0.5, CV_PI /2/180, 200);  // 输入的时二值图像，输出vector向量
    cout<<"检测出"<<lines.size()<<"条线"<<endl;
    for (size_t i=0; i < lines.size(); i++) {
        float rho = lines[i][0]; //就是圆的半径r
        float theta = lines[i][1]; //就是直线的角度
        Point pt1, pt2;
        double cos_theta = cos(theta), sin_theta = sin(theta);
        double x0 = cos_theta*rho, y0 = sin_theta*rho;
        double k = -cos_theta/sin_theta;
        double b = rho/sin_theta;
        cout<<"直线的k="<<k<<",b="<<b<<endl;
        pt1.x = cvRound(x0 + 1000 * (-sin_theta));
        pt1.y = cvRound(y0 + 1000 * (cos_theta));
        pt2.x = cvRound(x0 - 1000 * (-sin_theta));
        pt2.y = cvRound(y0 - 1000 * (cos_theta));
        line(img_src_color, pt1, pt2, Scalar(0, 0, 255), 1.5); //Scalar函数用于调节线段颜色，就是你想检测到的线段显示的是什么颜色
    }
    imshow("Hough_Statistics ", img_src_color);
    imwrite("/Users/keyanchen/Files/Code/LineDetection/LineDetection/6.jpg", img_src_color);

    // 创建后文的种子候选区域
    Mat seeds_lookup = Mat::zeros(img_src.rows, img_src.cols, CV_8UC1);
    // 概率霍夫变换
    vector<Vec4i> points;
    cvtColor(img_src, img_src_color, COLOR_GRAY2BGR);
    HoughLinesP(img_added, points, 1, CV_PI/2/180, 200, 0, 120);  // 输入的时二值图像，输出vector向量
    cout<<"检测出"<<points.size()<<"条线"<<endl;
    for (int i=0; i < points.size(); i++) {
        Point pt1(points[i][0], points[i][1]);
        Point pt2(points[i][2], points[i][3]);
        double k = float(pt2.y-pt1.y)/(pt2.x-pt1.x);
        double b = pt2.y-pt2.x/(pt2.x-pt1.x)*(pt2.y-pt1.y);
        cout<<"直线的k="<<k<<",b="<<b<<endl;
        line(img_src_color, pt1, pt2, cv::Scalar(0, 0, 255), 1.5);
        line(seeds_lookup, pt1, pt2, 255, 2);
    }
    imshow("Hough_Probability", img_src_color);
    imwrite("/Users/keyanchen/Files/Code/LineDetection/LineDetection/7.jpg", img_src_color);
    imshow("seeds_lookup", seeds_lookup);
    imwrite("/Users/keyanchen/Files/Code/LineDetection/LineDetection/8.jpg", seeds_lookup);

    // 求最长连通域
    queue<Coordinate> label_coordinate_queue;
    Mat label_mat = Mat::zeros(img_src.rows, img_src.cols, CV_8UC1);

    for(int row=0; row < img_src.rows; row++){
        for(int col=0; col<img_src.cols; col++){
            if(seeds_lookup.at<uchar>(row, col)==255 && img_added.at<uchar>(row, col)==255 && label_mat.at<uchar>(row, col)==0){
                label_coordinate_queue.push(Coordinate{row, col});
                while(!label_coordinate_queue.empty()){
                    SearchNeighbor(img_added, label_mat, label_coordinate_queue);
                }
            }
        }
    }
    imshow("Longest_Connected_Component_Label", label_mat);
    imwrite("/Users/keyanchen/Files/Code/LineDetection/LineDetection/9.jpg", label_mat);
    element = getStructuringElement(MORPH_ELLIPSE,Size(3,3));
    dilate(label_mat, label_mat, element);
    imwrite("/Users/keyanchen/Files/Code/LineDetection/LineDetection/10.jpg", label_mat);

    //插值
    Mat img_interpolation_median = img_src.clone();
    Mat img_interpolation_mean = img_src.clone();
    for(int row=0; row < img_src.rows; row++){
        for(int col=0; col<img_src.cols; col++){
            if(label_mat.at<uchar>(row, col)==255){
                Blur(img_interpolation_median, label_mat, row, col, "median", 3);
                Blur(img_interpolation_mean, label_mat, row, col, "mean", 3);
            }
        }
    }
    imshow("img_interpolation_median", img_interpolation_median);
    imwrite("/Users/keyanchen/Files/Code/LineDetection/LineDetection/11.jpg", img_interpolation_median);
    imshow("img_interpolation_mean", img_interpolation_mean);
    imwrite("/Users/keyanchen/Files/Code/LineDetection/LineDetection/12.jpg", img_interpolation_mean);

    waitKey(0);
    destroyAllWindows();
    return 0;
}
void Blur(Mat &img_interpolation, Mat label_mat, int row, int col, String pattern, int ksize=3){
    vector<uchar> data;
    int sum_data = 0;
    for(int i = row-floor(ksize/2); i<row+ceil(ksize/2); i++){
        for(int j = col-floor(ksize/2); j<col+ceil(ksize/2); j++){
            if(label_mat.at<uchar>(i, j)==0){
                data.push_back(img_interpolation.at<uchar>(i, j));
                sum_data += data.back();
            }
        }
    }
    sort(data.begin(), data.end());
//    vector<int> valid_data;
//    int sum_data = 0;
//    int mean_for_select_valid_data = sum(/9
//    for(int i=0;i<9;i++){
//        if(data[i] > 50){
//            valid_data.push_back(data[i]);
//            sum_data += data[i];
//        }
//    }
    if(data.empty()){
        Blur(img_interpolation, label_mat, row, col, pattern, ksize=ksize+2);
        return;
    }
    if(pattern=="median"){
        img_interpolation.at<uchar>(row, col) = data.at(floor(data.size()/2));
    }
    else if(pattern=="mean"){
        img_interpolation.at<uchar>(row, col) = uchar(int(sum_data/data.size()));
    }
}


void SearchNeighbor(Mat img_added, Mat &label_mat, queue<Coordinate> &label_coordinate_queue){
    Coordinate tempxy = label_coordinate_queue.front();
    label_coordinate_queue.pop();
    label_mat.at<uchar>(tempxy.x, tempxy.y) = 255;

    for(int k=0; k<8; k++){
        tempxy = {tempxy.x+NeighborDirection[k][0], tempxy.y+NeighborDirection[k][1]};
        if(tempxy.x<0 || tempxy.x>label_mat.rows || tempxy.y<0 || tempxy.y>label_mat.cols){
            continue;
        }
        if(img_added.at<uchar>(tempxy.x, tempxy.y)==255 && label_mat.at<uchar>(tempxy.x, tempxy.y)==0){
            label_coordinate_queue.push(tempxy);
        }
    }
}

//// 求最长连通域
//int length_best = 0;
//Mat label_mat_best;
//queue<Coordinate> label_coordinate_queue;
//Mat label_mat_temp = Mat::zeros(img_src.rows, img_src.cols, CV_8UC1);

//for(int row=0; row < img_src.rows; row++){
//    for(int col=0; col<img_src.cols; col++){
//        int coordinate_temp[4] = {img_src.rows, img_src.cols, 0, 0};
//        if(img_added.at<uchar>(row, col)==255 && label_mat_temp.at<uchar>(row, col)==0){
//            label_coordinate_queue.push(Coordinate{row, col});
//            while(!label_coordinate_queue.empty()){
//                SearchNeighbor(img_added, label_mat_temp, label_coordinate_queue, coordinate_temp);
//            }
//            int length_temp = max(coordinate_temp[2]-coordinate_temp[0], coordinate_temp[3]-coordinate_temp[1]);
//            //imshow("Longest_Connected_Component_Label_temp", label_mat_temp);
//            //waitKey(0);
//            cout<<length_temp<<endl;
//            if(length_temp>length_best){
//                length_best = length_temp;
//                label_mat_best = label_mat_temp.clone();
//            }
//            label_mat_temp = Mat::zeros(img_src.rows, img_src.cols, CV_8UC1);
//        }
//    }
//}
//imshow("Longest_Connected_Component_Label", label_mat_best);

//waitKey(0);
//destroyAllWindows();
//return 0;
//}

//void SearchNeighbor(Mat img_added, Mat &label_mat_temp, queue<Coordinate> &label_coordinate_queue, int coordinate_temp[4]){
//Coordinate tempxy = label_coordinate_queue.front();
//label_coordinate_queue.pop();
//if(tempxy.x<coordinate_temp[0]){
//    coordinate_temp[0] = tempxy.x;
//}
//if(tempxy.x>coordinate_temp[2]){
//    coordinate_temp[2] = tempxy.x;
//}
//if(tempxy.y<coordinate_temp[1]){
//    coordinate_temp[1] = tempxy.y;
//}
//if(tempxy.y>coordinate_temp[3]){
//    coordinate_temp[3] = tempxy.y;
//}
//label_mat_temp.at<uchar>(tempxy.x, tempxy.y) = 255;

//for(int k=0; k<14; k++){
//    tempxy = {tempxy.x+NeighborDirection[k][0], tempxy.y+NeighborDirection[k][1]};
//    if(tempxy.x<0 || tempxy.x>label_mat_temp.rows || tempxy.y<0 || tempxy.y>label_mat_temp.cols){
//        continue;
//    }
//    if(img_added.at<uchar>(tempxy.x, tempxy.y)==255 && label_mat_temp.at<uchar>(tempxy.x, tempxy.y)==0){
//        label_coordinate_queue.push(tempxy);
//    }
//}
//}
