#pragma once

#include <opencv2\core\core.hpp>
#include <opencv2\imgproc\imgproc.hpp>

#include "Methods.h"

using namespace cv;

void mySearch (Mat& output, Mat& dataImg, Mat& queryImg, Rect& queryRect, int* HR_exist, double hr_weight);