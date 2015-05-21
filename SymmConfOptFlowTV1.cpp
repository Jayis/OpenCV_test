#include "SymmConfOptFlowTV1.h"

SymmConfOptFlow_calc::SymmConfOptFlow_calc()
{
	//OptFlow = createOptFlow_DualTVL1();
	//OptFlow_back = createOptFlow_DualTVL1();
	OptFlow = new Mod_OpticalFlowDual_TVL1;
	OptFlow_back = new Mod_OpticalFlowDual_TVL1;
}

void SymmConfOptFlow_calc::calc(InputArray _I0, InputArray _I1, InputOutputArray _flow, InputOutputArray _flow_back, InputOutputArray _conf)
{
	OptFlow->calc_part1(_I0, _I1, _flow);
	OptFlow_back->calc_part1(_I1, _I0, _flow_back);

	// 
	nscales = OptFlow->getInt("nscales");
	confs.resize(nscales);
	flows.resize(nscales);
	flows_back.resize(nscales);

	// pyramidal structure for computing the optical flow
    for (int s = nscales - 1; s >= 0; --s)
    {
		cout << "s: " << int2str(s) << endl;

		cout << "procSpecScale" << endl;
        // compute the optical flow at the current scale
		OptFlow->procSpecScale(s);
		OptFlow_back->procSpecScale(s);

		cout << "getFlowSpecScale" << endl;
		Mat flow, flow_back, conf;
		OptFlow->getFlowSpecScale(s, flow);
		OptFlow_back->getFlowSpecScale(s, flow_back);
		showConfidence(flow, flow_back, conf);

		cout << "getInterpFlowSpecScale" << endl;
		Mat interp_flow, interp_flow_back, interp_conf;
		if (s < nscales - 1) {
			OptFlow->getInterpFlowSpecScale(s, interp_flow);
			OptFlow_back->getInterpFlowSpecScale(s, interp_flow_back);
			showConfidence(interp_flow, interp_flow_back, interp_conf);
		}
		else {
			interp_conf = -1 * Mat::ones(flow.rows, flow.cols, CV_64F);
		}

		// (Harry) before upscale the optical flow, select higher confidence form last layer
		cout << "combineFlow" << endl;
		flows[s] = Mat::zeros(flow.rows, flow.cols, CV_32FC2);
		flows_back[s] = Mat::zeros(flow_back.rows, flow_back.cols, CV_32FC2);
		confs[s] = Mat::zeros(flow.rows, flow.cols, CV_64F);
		for (int i = 0; i < conf.rows; i++) for (int j = 0; j < conf.cols; j++) {
			if (conf.at<double>(i, j) > interp_conf.at<double>(i, j)) {
				flows[s].at<Vec2f>(i, j) = flow.at<Vec2f>(i, j);
				flows_back[s].at<Vec2f>(i, j) = flow_back.at<Vec2f>(i, j);
				confs[s].at<double>(i, j) = conf.at<double>(i, j);
			}
			else {
				flows[s].at<Vec2f>(i, j) = interp_flow.at<Vec2f>(i, j);
				flows_back[s].at<Vec2f>(i, j) = interp_flow_back.at<Vec2f>(i, j);
				confs[s].at<double>(i, j) = interp_conf.at<double>(i, j);
			}
		}

        // if this was the last scale, finish now
        if (s == 0)
            break;

		cout << "setFlowForNextScale" << endl;
        // otherwise, upsample the optical flow
		OptFlow->setFlowForNextScale(s, flows[s]);
		OptFlow_back->setFlowForNextScale(s, flows_back[s]);
    }

    //Mat uxy[] = {u1s[0], u2s[0]};
    //merge(uxy, 2, _flow);
	flows[0].copyTo(_flow);
	flows_back[0].copyTo(_flow_back);
	confs[0].copyTo(_conf);
}