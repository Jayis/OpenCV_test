#pragma once

#include <opencv2\video\video.hpp>

#include "Mod_tv1flow.h"
#include "Methods.h"

using namespace cv;

class symmConfOptFlow_calc
{
public:
	symmConfOptFlow_calc();

	void calc(InputArray I0, InputArray I1, InputOutputArray flow, InputOutputArray flow_back);

private:
	Mod_OpticalFlowDual_TVL1 *OptFlow, *OptFlow_back;

	int nscales;

	vector<Mat> confs, flows;
};