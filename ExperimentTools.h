#pragma once

#include <opencv2\core\core.hpp>

using namespace cv;
using namespace std;

void NaiveForwardNNWarp (Mat& source, Mat& flow, Mat& output, int ch);