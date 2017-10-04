
#include <iostream>
//#include "opencv2/opencv_modules.hpp"
//#include <time.h>
#include <sys/time.h>
#include <opencv2/imgproc.hpp>
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/highgui.hpp"
//#include "opencv2/cudafeatures2d.hpp"
//#include "opencv2/xfeatures2d/cuda.hpp"

#include "opencv2/calib3d.hpp"

using namespace std;
using namespace cv;
//using namespace cv::cuda;


int main () {
    cv::Mat img(512, 512, CV_8UC3, cv::Scalar(0));

    cv::putText(img, "Hello, world", cv::Point(10, img.rows/2), cv::FONT_HERSHEY_DUPLEX, 1.0, CV_RGB(118, 185, 0), 2);

    cv::imshow("show", img);
    cv::waitKey();
}
//
//static void help()
//{
//    cout << "\nThis program demonstrates using SURF_CUDA features detector, descriptor extractor and BruteForceMatcher_CUDA" << endl;
//    cout << "\nUsage:\n\tsurf_keypoint_matcher --left <image1> --right <image2>" << endl;
//}
//
//
//
//typedef unsigned long long timestamp_t;
//
//static timestamp_t get_timestamp(){
//    struct timeval now;
//    gettimeofday (&now, NULL);
//    return  now.tv_usec + (timestamp_t)now.tv_sec * 1000000;
//}
//
//
//
//
//
//int main(int argc, char* argv[])
//{
//    string log_file = "/home/benwang/code/c3_prototyping/c3.scan.shipside.new.app/test/homography_test/log_1505423546.75_13_0.0738260746002.png";
//      string ticket_file = "/home/benwang/code/c3_prototyping/c3.scan.shipside.new.app/test/2017-06-28T15:03:34+1200/1498619026.45.png";
////      string ticket_file = "/home/benwang/code/c3_prototyping/c3.scan.shipside.new.app/test/2017-06-28T15:03:34+1200/1498619024.94.png";
////    string ticket_file = "/home/benwang/code/c3_prototyping/c3.scan.shipside.new.app/test/2017-06-28T15:03:34+1200/1498619018.23.png";
//    string test3 = "/home/benwang/code/c3_prototyping/c3.scan.shipside.new.app/test/2017-06-28T15:03:34+1200/1498619024.94.png";
//    string test4 = "/home/benwang/code/c3_prototyping/c3.scan.shipside.new.app/test/2017-06-28T15:03:34+1200/1498619018.23.png";
//
//    string test5 = "/home/benwang/code/c3_prototyping/c3.scan.shipside.new.app/test/2017-06-28T15:03:34+1200/1498619019.02.png";
//    string test6 = "/home/benwang/code/c3_prototyping/c3.scan.shipside.new.app/test/2017-06-28T15:03:34+1200/1498619020.49.png";
//
//
//
//    Mat image1 = imread(ticket_file, IMREAD_GRAYSCALE);
//    Mat image2 = imread(log_file, IMREAD_GRAYSCALE);
//
//
//    Mat image3 = imread(test3, IMREAD_GRAYSCALE);
//    Mat image4 = imread(test4, IMREAD_GRAYSCALE);
//
//    Mat image5 = imread(test5, IMREAD_GRAYSCALE);
//    Mat image6 = imread(test6, IMREAD_GRAYSCALE);
//
//    resize(image1, image1, Size(), 0.5, 0.5, INTER_CUBIC);
////    resize(image2, image2, Size(), 0.5, 0.5, INTER_CUBIC);
//
//    timestamp_t t0 = get_timestamp();
//
//    GpuMat img1, img2;
//    img1.upload(image1);
//    timestamp_t t11 = get_timestamp();
//    img2.upload(image2);
//    timestamp_t t12 = get_timestamp();
//
//    GpuMat img3, img4;
//    img3.upload(image3);
//    timestamp_t t13 = get_timestamp();
//    img4.upload(image4);
//    timestamp_t t14 = get_timestamp();
//
//    GpuMat img5, img6;
//    img3.upload(image5);
//    timestamp_t t15 = get_timestamp();
//    img4.upload(image6);
//    timestamp_t t16 = get_timestamp();
//
//    cout << "Test the upload time: "
//         << (t11 - t0) / 1000000.0L << "  " <<
//           (t12 - t11) / 1000000.0L <<  "  " <<
//          (t13 - t12) / 1000000.0L << "  " <<
//                                   (t14 - t13) / 1000000.0L <<  "  " <<
//                                                           (t15 - t14) / 1000000.0L  << "  " <<
//                                                                                    (t16 - t15) / 1000000.0L <<" seconds" << endl;
//
//    timestamp_t t1 = get_timestamp();
//    double secs = (t1 - t0) / 1000000.0L;
////    cout << "Test the upload time: " << secs << " seconds" << endl;
//
//    cv::cuda::printShortCudaDeviceInfo(cv::cuda::getDevice());
//
//    SURF_CUDA surf;
//
//    // detecting keypoints & computing descriptors
//    GpuMat keypoints1GPU, keypoints2GPU;
//    GpuMat descriptors1GPU, descriptors2GPU;
//    surf(img1, GpuMat(), keypoints1GPU, descriptors1GPU);
//    surf(img2, GpuMat(), keypoints2GPU, descriptors2GPU);
//
//    GpuMat keypoints3GPU, keypoints4GPU, keypoints5GPU, keypoints6GPU;
//    GpuMat descriptors3GPU, descriptors4GPU,descriptors5GPU, descriptors6GPU;
//    surf(img3, GpuMat(), keypoints3GPU, descriptors3GPU);
//    surf(img4, GpuMat(), keypoints4GPU, descriptors4GPU);
//    surf(img5, GpuMat(), keypoints5GPU, descriptors5GPU);
//    surf(img6, GpuMat(), keypoints6GPU, descriptors6GPU);
//
//    timestamp_t t2 = get_timestamp();
//    secs = (t2 - t1) / 1000000.0L;
//    cout << "Test the feature time: " << secs << " seconds" << endl;
//
//    cout << "FOUND " << keypoints1GPU.cols << " keypoints on first image" << endl;
//    cout << "FOUND " << keypoints2GPU.cols << " keypoints on second image" << endl;
//
//    // matching descriptors
//    // !!!!!!!! Here is BFMatcher not FlannBaseMatcher !!!!!!!!!!!
//    Ptr<cv::cuda::DescriptorMatcher> matcher = cv::cuda::DescriptorMatcher::createBFMatcher(surf.defaultNorm());
//    vector<vector<DMatch>> matches;
//    matcher->knnMatch(descriptors1GPU, descriptors2GPU, matches, 2);
//
//    matcher->knnMatch(descriptors1GPU, descriptors3GPU, matches, 2);
//    matcher->knnMatch(descriptors1GPU, descriptors4GPU, matches, 2);
//    matcher->knnMatch(descriptors1GPU, descriptors5GPU, matches, 2);
//    matcher->knnMatch(descriptors1GPU, descriptors6GPU, matches, 2);
//
//
//    timestamp_t t3 = get_timestamp();
//    secs = (t3 - t2) / 1000000.0L;
//    cout << "Test the knnMatch time: " << secs << " seconds" << endl;
//
//
//    // downloading results
//    vector<KeyPoint> keypoints1, keypoints2;
//    vector<float> descriptors1, descriptors2;
//    surf.downloadKeypoints(keypoints1GPU, keypoints1);
//    surf.downloadKeypoints(keypoints2GPU, keypoints2);
//    surf.downloadDescriptors(descriptors1GPU, descriptors1);
//    surf.downloadDescriptors(descriptors2GPU, descriptors2);
//
//    timestamp_t t4 = get_timestamp();
//    secs = (t4 - t3) / 1000000.0L;
//    cout << "Test the download time: " << secs << " seconds" << endl;
//
//
//    ///////////////  code below is from sample_surf_knnMatch.cpp /////////////
//
//    std::vector< DMatch > good_matches;
//
//    for (auto i: matches) {
//
//        if (i[0].distance < 0.75 * i[1].distance) {
//            good_matches.push_back(i[0]);
//        }
//    }
//
//    Mat img_matches;
//    Mat img_object = Mat(img1);
//    Mat img_scene = Mat(img2);
//    vector<KeyPoint> keypoints_object = keypoints1;
//    vector<KeyPoint> keypoints_scene = keypoints2;
//
//
//    timestamp_t t5 = get_timestamp();
//    secs = (t5 - t4) / 1000000.0L;
//    cout << "Test the other time: " << secs << " seconds" << endl;
//
//
//
//    drawMatches( img_object, keypoints_object, img_scene, keypoints_scene,
//               good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
//               std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
//    //-- Localize the object
//    std::vector<Point2f> obj;
//    std::vector<Point2f> scene;
//    for( size_t i = 0; i < good_matches.size(); i++ )
//    {
//        //-- Get the keypoints from the good matches
//        obj.push_back( keypoints_object[ good_matches[i].queryIdx ].pt );
//        scene.push_back( keypoints_scene[ good_matches[i].trainIdx ].pt );
//    }
//    Mat H = findHomography( obj, scene, RANSAC );
//    //-- Get the corners from the image_1 ( the object to be "detected" )
//    std::vector<Point2f> obj_corners(4);
//    obj_corners[0] = cvPoint(0,0); obj_corners[1] = cvPoint( img_object.cols, 0 );
//    obj_corners[2] = cvPoint( img_object.cols, img_object.rows ); obj_corners[3] = cvPoint( 0, img_object.rows );
//    std::vector<Point2f> scene_corners(4);
//    perspectiveTransform( obj_corners, scene_corners, H);
//    //-- Draw lines between the corners (the mapped object in the scene - image_2 )
//    line( img_matches, scene_corners[0] + Point2f( img_object.cols, 0), scene_corners[1] + Point2f( img_object.cols, 0), Scalar(0, 255, 0), 4 );
//    line( img_matches, scene_corners[1] + Point2f( img_object.cols, 0), scene_corners[2] + Point2f( img_object.cols, 0), Scalar( 0, 255, 0), 4 );
//    line( img_matches, scene_corners[2] + Point2f( img_object.cols, 0), scene_corners[3] + Point2f( img_object.cols, 0), Scalar( 0, 255, 0), 4 );
//    line( img_matches, scene_corners[3] + Point2f( img_object.cols, 0), scene_corners[0] + Point2f( img_object.cols, 0), Scalar( 0, 255, 0), 4 );
//
//
//    secs = (t5 - t0) / 1000000.0L;
//    cout << "All the time: " << secs << " seconds" << endl;
//    //-- Show detected matches
//    imshow( "Good Matches & Object detection", img_matches );
//
//    waitKey(0);
//
//// reference: https://morf.lv/complete-guide-using-surf-feature-detector
//
//    return 0;
//}