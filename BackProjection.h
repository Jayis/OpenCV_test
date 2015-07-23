#pragma once

#include <vector>
#include <iostream>
#include <string>
#include <sstream>

#include <opencv2\core\core.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\highgui\highgui.hpp>

#include "DataStructures.h"
#include "Methods.h"
#include "Macros.h"

//using namespace cv;
using namespace std;

void BackProjection ( Mat& HRimg, double scale, vector<Mat>& imgs, vector<Mat>& flows, Mat& PSF, Mat& BPkernel, TermCriteria& BPstop );

void BackProjection_Confidence ( Mat& HRimg, double scale, vector<Mat>& imgs, vector<Mat>& flows, Mat& PSF, Mat& BPkernel, TermCriteria& BPstop,  vector<Mat>& confidences );


