#pragma once

#include <string>
#include <sstream>

#include <opencv2/core/core.hpp>
#include <opencv2\highgui\highgui.hpp>

using namespace std;
using namespace cv;

string int2str(int i);

void writeImgDiff(Mat& a, Mat& b, string& name);