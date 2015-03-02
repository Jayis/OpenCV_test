#pragma once

#include <iostream>

#include <opencv2\core\core.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\photo\photo.hpp>
#include <opencv2\highgui\highgui.hpp>

#include "Tools.h"

using namespace std;
using namespace cv;

void penalty (vector<Mat>& y, Mat& x_bar, double gamma);

void data_fidelity (Mat& x_k1, Mat& x_k, vector<Mat>& y, double tau);

