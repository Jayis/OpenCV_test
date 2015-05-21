#pragma once

#include "opencv2/video/tracking.hpp"
#include "opencv2/video/background_segm.hpp"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/core/internal.hpp"
#include "opencv2/core/core.hpp"

#include <list>
#include <iostream>

using namespace std;
using namespace cv;

class Mod_OpticalFlowDual_TVL1 : public DenseOpticalFlow
{
public:
    Mod_OpticalFlowDual_TVL1();

    void calc(InputArray I0, InputArray I1, InputOutputArray flow);
	void collectGarbage();

	void calc_part1(InputArray I0, InputArray I1, InputOutputArray flow);
	void calc_part2(InputOutputArray flow);
    void procSpecScale(int s);
	void getFlowSpecScale(int s, Mat& output);
	void getInterpFlowSpecScale(int s, Mat& output);

    AlgorithmInfo* info() const;

protected:
    double tau;
    double lambda;
    double theta;
    int nscales;
    int warps;
    double epsilon;
    int iterations;
    bool useInitialFlow;

private:
    void procOneScale(const Mat_<float>& I0, const Mat_<float>& I1, Mat_<float>& u1, Mat_<float>& u2);

    std::vector<Mat_<float> > I0s;
    std::vector<Mat_<float> > I1s;
    std::vector<Mat_<float> > u1s;
    std::vector<Mat_<float> > u2s;

    Mat_<float> I1x_buf;
    Mat_<float> I1y_buf;

    Mat_<float> flowMap1_buf;
    Mat_<float> flowMap2_buf;

    Mat_<float> I1w_buf;
    Mat_<float> I1wx_buf;
    Mat_<float> I1wy_buf;

    Mat_<float> grad_buf;
    Mat_<float> rho_c_buf;

    Mat_<float> v1_buf;
    Mat_<float> v2_buf;

    Mat_<float> p11_buf;
    Mat_<float> p12_buf;
    Mat_<float> p21_buf;
    Mat_<float> p22_buf;

    Mat_<float> div_p1_buf;
    Mat_<float> div_p2_buf;

    Mat_<float> u1x_buf;
    Mat_<float> u1y_buf;
    Mat_<float> u2x_buf;
    Mat_<float> u2y_buf;
};