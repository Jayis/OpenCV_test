#pragma once

#include <opencv2\video\video.hpp>

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
	Mod_OpticalFlowDual_TVL1 *OptFlow, *OptFlow_back;

	int nscales;

	vector<Mat> confs, flows, flows_back;
	Mat fromWhichLayer;
};