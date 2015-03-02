#pragma once

#include <opencv2\core\core.hpp>
#include <opencv2\imgproc\imgproc.hpp>

#include "Macros.h"

using namespace cv;
using namespace std;

void NaiveForwardNNWarp (Mat& source, Mat& flow, Mat& output, int ch);

void showConfidence (Mat& flow_forward, Mat& flow_backward, Mat& confidence);

double ExpNegSQR (float x);

double calcConfidence (Vec2f& f, Vec2f& b);

void HR_to_LR ( Mat& HRimg, Mat& LRimg, double scale, Mat& PSF, bool super_PSF, double PSF_scale=0 );

void preInterpolation ( Mat& PSF, Mat& super_PSF, double PSF_scale );