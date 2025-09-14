///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2024, STEREOLABS.
//
// All rights reserved.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
///////////////////////////////////////////////////////////////////////////

/************************************************************
** This sample demonstrates how to read a SVO video file. **
** We use OpenCV to display the video.					   **
*************************************************************/

// ZED include
#include <sl/Camera.hpp>

// Sample includes
#include <opencv2/opencv.hpp>
#include "utils.hpp"
#include <iostream>
#include "../../../../../CMake_builds_ZED_Camera_Project/Filter.h"

// Using namespace
using namespace sl;
using namespace std;


void print(string msg_prefix, ERROR_CODE err_code = ERROR_CODE::SUCCESS, string msg_suffix = "");

int main(int argc, char** argv) {


    // Create ZED objects
    Camera zed;
    InitParameters init_parameters;
    init_parameters.input.setFromSVOFile("svo_files/file1.svo");
    init_parameters.depth_mode = sl::DEPTH_MODE::NEURAL;

    // Open the camera
    auto returned_state = zed.open(init_parameters);
    if (returned_state != ERROR_CODE::SUCCESS) {
        print("Camera Open", returned_state, "Exit program.");
        return EXIT_FAILURE;
    }

    std::string s;
    for (const auto& piece : zed.getSVODataKeys()) s += piece + "; ";
    std::cout << "Channels that are in the SVO: " << s << std::endl;

    unsigned long long last_timestamp_ns;

    std::map<sl::Timestamp, sl::SVOData> data_map;
    std::cout << "Reading everything all at once." << std::endl;
    auto ing = zed.retrieveSVOData("TEST", data_map);

    for (const auto& d : data_map) {
        std::string s;
        d.second.getContent(s);
        std::cout << d.first << " (//) " << s << std::endl;
    }

    std::cout << "#########\n";

    auto resolution = zed.getCameraInformation().camera_configuration.resolution; \

	// Zed matrix for color image
    Mat colorImage(resolution, MAT_TYPE::U8_C4, MEM::CPU);
	// OpenCV matrix for color image
    cv::Mat colorImage_ocv = slMat2cvMat(colorImage);
	// the data is stored as Vec4b (4 channels of unsigned char), so colorData is a pointer to Vec4b
    cv::Vec4b* colorData = (cv::Vec4b*)colorImage_ocv.data;

	// a new matrix OpenCV (on four channels (RGBA) unsigned char ....that is in the range 0 ..255)
    cv::Mat colorImageProcessed_ocv = cv::Mat(resolution.height, resolution.width, CV_8UC4);
	// the processed data will be stored here, colorProcessedData is a pointer to Vec4b
    cv::Vec4b* colorProcessedData = (cv::Vec4b*)colorImageProcessed_ocv.data;


    Mat depthImage(resolution, MAT_TYPE::U8_C4, MEM::CPU);
    cv::Mat depthImage_ocv = slMat2cvMat(depthImage);
    cv::Vec4b* depthData = (cv::Vec4b*)depthImage_ocv.data;

    cv::Mat depthProcessed_ocv = cv::Mat(resolution.height, resolution.width, CV_8UC4);

    cv::Vec4b* depthProcessedData = (cv::Vec4b*)depthProcessed_ocv.data;


    cv::Mat grayscaleImageProcessed_ocv = cv::Mat(resolution.height, resolution.width, CV_8UC1);
    uchar* grayscaleProcessedData = (uchar*)grayscaleImageProcessed_ocv.data;

    Mat grayscaleImage(resolution, MAT_TYPE::U8_C1, MEM::CPU);
    cv::Mat grayscaleImage_ocv = slMat2cvMat(grayscaleImage);
    uchar* grayscaleData = (uchar*)grayscaleImage_ocv.data;

    Mat normalImage(resolution, MAT_TYPE::U8_C4, MEM::CPU);
    cv::Mat normalImage_ocv = slMat2cvMat(normalImage);
    uchar* normalData = (uchar*)normalImage_ocv.data;


    Mat normalMeasure(resolution, MAT_TYPE::F32_C4, MEM::CPU);
    cv::Mat normalMeasure_ocv = slMat2cvMat(normalMeasure);
    cv::Vec4f* normalMeasureData = (cv::Vec4f*)normalMeasure_ocv.data;

    Mat normalImageComputed(resolution, MAT_TYPE::U8_C4, MEM::CPU);
    cv::Mat normalImageComputed_ocv = slMat2cvMat(normalImageComputed);
    cv::Vec4b* normalImageComputedData = (cv::Vec4b*)normalImageComputed_ocv.data;


    Mat normalMeasureComputed(resolution, MAT_TYPE::F32_C4, MEM::CPU);
    cv::Mat normalMeasureComputed_ocv = slMat2cvMat(normalMeasureComputed);
    cv::Vec4f* normalMeasureComputedData = (cv::Vec4f*)normalMeasureComputed_ocv.data;


    Mat pointCloud(resolution, MAT_TYPE::F32_C4, MEM::CPU);
    cv::Mat pointCloud_ocv = slMat2cvMat(pointCloud);
    cv::Vec4f* pointCloudData = (cv::Vec4f*)pointCloud_ocv.data;




	// For task 3 -> BFS segmentation with seed points, I'll initialize a buffer.
	// I create an openCV matrix for displaying the segmentation result.
    cv::Mat segImg(resolution.height, resolution.width, CV_8UC4);
    cv::Vec4b* segmentationImageOutput = (cv::Vec4b*)segImg.data;



    // Setup key, images, times
    char key = ' ';
    last_timestamp_ns = zed.getTimestamp(sl::TIME_REFERENCE::IMAGE);
    while (key != 'q') {
        returned_state = zed.grab();
        if (returned_state <= ERROR_CODE::SUCCESS) {
            std::map<sl::Timestamp, sl::SVOData> data_map;
            //std::cout << "Reading between "<< last_timestamp_ns << " and " << zed.getTimestamp(sl::TIME_REFERENCE::IMAGE) << std::endl;
            auto ing = zed.retrieveSVOData("TEST", data_map, last_timestamp_ns, zed.getTimestamp(sl::TIME_REFERENCE::IMAGE));
            zed.retrieveImage(colorImage, VIEW::RIGHT, MEM::CPU, resolution);
            zed.retrieveImage(depthImage, VIEW::DEPTH, MEM::CPU, resolution);
            zed.retrieveImage(grayscaleImage, VIEW::LEFT_GRAY, MEM::CPU, resolution);
            zed.retrieveImage(normalImage, VIEW::NORMALS, MEM::CPU, resolution);

            zed.retrieveMeasure(normalMeasure, MEASURE::NORMALS, MEM::CPU, resolution);
            zed.retrieveMeasure(pointCloud, MEASURE::XYZ, MEM::CPU, resolution);


            // Some calls to the functions from the first task.

            Filter::filterGrayscaleMedianFilter(grayscaleData, grayscaleProcessedData, (int)resolution.width, (int)resolution.height);
            Filter::filterGrayscaleSobel(grayscaleData, grayscaleProcessedData, (int)resolution.width, (int)resolution.height);
            Filter::filterColorAverage(colorData, colorProcessedData, (int)resolution.width, (int)resolution.height);
            //Filter::filterDepthGaussian(depthData, depthProcessedData, (int)resolution.width, (int)resolution.height);
            Filter::filterDepthPrewitt(depthData, depthProcessedData, (int)resolution.width, (int)resolution.height);
            //Filter::colorToGrayscale(colorImage_ocv);
            //Filter::filterColorAverage(colorData, colorProcessedData, (int)resolution.width, (int)resolution.height);

            Filter::computeNormals(pointCloudData, normalMeasureComputedData, (int)resolution.width, (int)resolution.height);
            Filter::transformNormalsToImage(normalMeasureComputedData, normalImageComputedData, (int)resolution.width, (int)resolution.height);

			// Testing the functionality of the Filter last 2 functions.
            Filter::computeNormals5x5Vicinity(pointCloudData, normalMeasureComputedData, (int)resolution.width, (int)resolution.height);
            Filter::filterNormalSobel(normalMeasureComputedData, normalImageComputedData, (int)resolution.width, (int)resolution.height);


			// TASK 3: BFS segmentation with seed points
			std::vector<int> labels; // for saving the labels of each pixel.
			std::vector<Filter::RegionSummary> regions; // for saving the statistics of each region.

            Filter::segmentBFSFromSeeds(
                grayscaleData,
                pointCloudData,
                normalMeasureData,
                (int)resolution.width,
                (int)resolution.height,
                40,    // numSeeds
                0.7f,  // w1 (I)
                0.3f,  // w2 (D)
                0.2f,  // w3 (N) 
                0.01f, // treshold.
                labels,
                regions
            );

            Filter::labelsToColorImage(labels, (int)resolution.width, (int)resolution.height, segmentationImageOutput);

			// Display results.
            cv::imshow("Color", colorImage_ocv);
            cv::imshow("Depth", depthImage_ocv);
            cv::imshow("Grayscale", grayscaleImage_ocv);
            cv::imshow("Normal", normalImage_ocv);
            cv::imshow("Normal image computed", normalImageComputed_ocv);

            cv::imshow("Segmentation BFS", segImg); // For task 3.

            key = cv::waitKey(10);
        }
        else if (returned_state == sl::ERROR_CODE::END_OF_SVOFILE_REACHED) {
            print("SVO end has been reached. Looping back to 0\n");
            zed.setSVOPosition(0);
        }
        else {
            print("Grab ZED : ", returned_state);
            break;
        }
        last_timestamp_ns = zed.getTimestamp(sl::TIME_REFERENCE::IMAGE);
    }
    zed.close();
    return EXIT_SUCCESS;
}

void print(string msg_prefix, ERROR_CODE err_code, string msg_suffix) {
    cout << "[Sample]";
    if (err_code != ERROR_CODE::SUCCESS)
        cout << "[Error] ";
    else
        cout << " ";
    cout << msg_prefix << " ";
    if (err_code != ERROR_CODE::SUCCESS) {
        cout << " | " << toString(err_code) << " : ";
        cout << toVerbose(err_code);
    }
    if (!msg_suffix.empty())
        cout << " " << msg_suffix;
    cout << endl;
}
