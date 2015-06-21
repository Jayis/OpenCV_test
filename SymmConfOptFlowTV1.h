#pragma once

#include <opencv2\video\video.hpp>

#include <vector>

#include "Mod_tv1flow.h"
#include "Methods.h"

using namespace cv;

class SymmConfOptFlow_calc
{
public:
	SymmConfOptFlow_calc();
	~SymmConfOptFlow_calc();

	void calc(InputArray I0, InputArray I1, InputOutputArray flow, InputOutputArray flow_back, InputOutputArray conf);

private:
	void selectHigherConf(Mat& flow, Mat& interp_flow, Mat& conf, Mat& interp_conf, Mat& combined_flow, Mat& combined_conf);
	void fillLowConf_WithH(Mat& curI0, Mat& flow, Mat& interp_flow, Mat& conf, Mat& interp_conf, Mat& combined_flow, Mat& combined_conf);

	Mod_OpticalFlowDual_TVL1 *OptFlow, *OptFlow_back;

	int nscales, s;

	vector<Mat> confs, flows, flows_back, confs_back;
	Mat fromWhichLayer;
};