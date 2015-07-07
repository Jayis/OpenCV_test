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

	void calc_tv1_exp(InputArray I0, InputArray I1, InputOutputArray flow, InputOutputArray flow_back, InputOutputArray conf);
	void calc_tv1(InputArray I0, InputArray I1, InputOutputArray flow, InputOutputArray flow_back, InputOutputArray conf);
	void calc_HS(InputArray I0, InputArray I1, InputOutputArray flow, InputOutputArray flow_back, InputOutputArray conf);
	void calc_SF(InputArray I0, InputArray I1, InputOutputArray flow, InputOutputArray flow_back, InputOutputArray conf);
	void calc_FB(InputArray I0, InputArray I1, InputOutputArray flow, InputOutputArray flow_back, InputOutputArray conf);

private:
	void selectHigherConf(Mod_OpticalFlowDual_TVL1* curOptFlow, Mat& flow, Mat& interp_flow, Mat& conf, Mat& interp_conf, Mat& combined_flow, Mat& combined_conf);
	void propagateFlow_WithH(Mod_OpticalFlowDual_TVL1* curOptFlow, Mat& flow, Mat& interp_flow, Mat& conf, Mat& interp_conf, Mat& combined_flow, Mat& combined_conf);
	void fillLowConf_WithH(Mod_OpticalFlowDual_TVL1* curOptFlow, Mat& flow, Mat& interp_flow, Mat& conf, Mat& interp_conf, Mat& combined_flow, Mat& combined_conf);
	void fillLowConf_WithLaplaceEQ(Mod_OpticalFlowDual_TVL1* curOptFlow, Mat& flow, Mat& interp_flow, Mat& conf, Mat& interp_conf, Mat& combined_flow, Mat& combined_conf);
	void patchI0_whileLowConf(Mod_OpticalFlowDual_TVL1* curOptFlow, Mat& flow, Mat& interp_flow, Mat& conf, Mat& interp_conf, Mat& combined_flow, Mat& combined_conf);

	Mod_OpticalFlowDual_TVL1 *OptFlow, *OptFlow_back;

	int nscales, s;

	vector<Mat> confs, flows, flows_back, confs_back;
	Mat fromWhichLayer;
};