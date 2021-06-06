#include <fstream>
#include <iostream>
#include <vector>
#include <random>
#include <opencv2/opencv.hpp>

int main(int argc, char** argv) {

    if(argc < 3) {
        std::cerr << "Not enough parameters!" << std::endl;
        return 1;
    }

    std::ifstream file(argv[1]);
    if(!file.is_open()) {
        std::cerr << "No such file: " << argv[1] << std::endl;
    }
    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_int_distribution<std::mt19937::result_type> dist_rot(0,40);

    float rot_angle = dist_rot(rng);
    rot_angle -= 20;
    std::cout << "z: " << rot_angle << " " << rot_angle * M_PI / 180 << std::endl;
    float cos_angle = std::cos(rot_angle * M_PI / 180);
    float sin_angle = std::sin(rot_angle * M_PI / 180);

    cv::Mat Rotation_Z = (cv::Mat_<float>(3,3) << cos_angle, -sin_angle, 0,
                                                  sin_angle, cos_angle, 0,
                                                  0, 0, 1);

    rot_angle = dist_rot(rng);
    rot_angle -= 20;
    std::cout << "x: " << rot_angle << " " << rot_angle * M_PI / 180 << std::endl;
    cos_angle = std::cos(rot_angle * M_PI / 180);
    sin_angle = std::sin(rot_angle * M_PI / 180);

    cv::Mat Rotation_X = (cv::Mat_<float>(3,3) << 1, 0, 0,
                                                  0, cos_angle, -sin_angle,
                                                  0, sin_angle, cos_angle);

    rot_angle = dist_rot(rng);
    rot_angle -= 20;
    std::cout << "y: " << rot_angle << " " << rot_angle * M_PI / 180 << std::endl;
    cos_angle = std::cos(rot_angle * M_PI / 180);
    sin_angle = std::sin(rot_angle * M_PI / 180);

    cv::Mat Rotation_Y = (cv::Mat_<float>(3,3) << cos_angle, 0, sin_angle,
                                                  0, 1, 0,
                                                  -sin_angle, 0, cos_angle);

    cv::Mat Rotation = Rotation_X * Rotation_Y * Rotation_Z;

    std::uniform_int_distribution<std::mt19937::result_type> dist_trans(0,10);

    cv::Mat translation(3,1, CV_32F);
    translation.at<float>(0,0) = dist_trans(rng);
    translation.at<float>(0,0) -= 5;
    translation.at<float>(1,0) = dist_trans(rng);
    translation.at<float>(1,0) -= 5;
    translation.at<float>(2,0) = dist_trans(rng);
    translation.at<float>(2,0) -= 5;
    std::cout << "trans: " << translation << std::endl;
    
    std::string output_filename = argv[2];
    std::ofstream out_file(output_filename + ".xyz");
    std::vector<cv::Mat> points;
    cv::Mat mean = (cv::Mat_<float>(3,1) << 0, 0, 0);
    
    float coord;
    while(file >> coord) {
        cv::Mat point(3,1, CV_32F);
        point.at<float>(0,0) = coord;
        file >> coord;
        point.at<float>(1,0) = coord;
        file >> coord;
        point.at<float>(2,0) = coord;
        mean += point;
        
        points.push_back(point.clone());
        // point = Rotation_X * point;
        // point += translation;

        // out_file << point.at<float>(0,0) << " " << point.at<float>(1,0) << " " << point.at<float>(2,0) << std::endl;
    }

    mean /= points.size();

    for(int i = 0; i < points.size(); ++i) {
        cv::Mat point = points[i].clone();
        point -= mean;
        point = Rotation * point;
        point += translation;
        point+=mean;
        out_file << point.at<float>(0,0) << " " << point.at<float>(1,0) << " " << point.at<float>(2,0) << std::endl;
    }

    std::ofstream out_rot_trans(output_filename + "_rot_trans.txt");
    out_rot_trans << translation.at<float>(0,0) << " " << translation.at<float>(1,0) << " " << translation.at<float>(2,0) << std::endl;
    for(int i = 0; i < 3; ++i) {
        for(int j = 0; j < 3; ++j) {
            out_rot_trans << Rotation.at<float>(i,j) << " ";
        }
        out_rot_trans << std::endl;
    }
    // cv::Mat rotvec(3,1,CV_32F) ;
    // cv::Rodrigues(Rotation*Rotation.t(), rotvec);
    // std::cout << rotvec << std::endl;
    // std::cout << "transation:" << std::endl;
    // std::cout << translation << std::endl;
    // std::cout << "rotation:" << std::endl;
    std::cout << Rotation << std::endl;
    return 0;
}


