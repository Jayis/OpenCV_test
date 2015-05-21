#include "SymmConfOptFlowTV1.h"

symmConfOptFlow_calc::symmConfOptFlow_calc()
{
	//OptFlow = createOptFlow_DualTVL1();
	//OptFlow_back = createOptFlow_DualTVL1();
	OptFlow = new Mod_OpticalFlowDual_TVL1;
	OptFlow_back = new Mod_OpticalFlowDual_TVL1;
}

void symmConfOptFlow_calc::calc(InputArray _I0, InputArray _I1, InputOutputArray _flow, InputOutputArray _flow_back)
{
	OptFlow->calc_part1(_I0, _I1, _flow);
	OptFlow_back->calc_part1(_I1, _I0, _flow_back);

	// 
	nscales = OptFlow->getInt("nscale");
	confs.resize(nscales);
	flows.resize(nscales);

	// pyramidal structure for computing the optical flow
    for (int s = nscales - 1; s >= 0; --s)
    {
        // compute the optical flow at the current scale
		OptFlow->procSpecScale(s);
		OptFlow_back->procSpecScale(s);

		Mat flow, flow_back, conf;
		OptFlow->getFlowSpecScale(s, flow);
		OptFlow_back->getInterpFlowSpecScale(s, flow_back);
		showConfidence(flow, flow_back, conf);

		Mat interp_flow, interp_flow_back, interp_conf;
		OptFlow->getInterpFlowSpecScale(s, interp_flow);
		OptFlow_back->getInterpFlowSpecScale(s, interp_flow_back);
		showConfidence(interp_flow, interp_flow_back, interp_conf);



        // if this was the last scale, finish now
        if (s == 0)
            break;

		// (Harry) before upscale the optical flow, select higher confidence form last layer

        // otherwise, upsample the optical flow
		/*
        // zoom the optical flow for the next finer scale
        resize(u1s[s], u1s[s - 1], I0s[s - 1].size());
        resize(u2s[s], u2s[s - 1], I0s[s - 1].size());

        // scale the optical flow with the appropriate zoom factor
        multiply(u1s[s - 1], Scalar::all(2), u1s[s - 1]);
        multiply(u2s[s - 1], Scalar::all(2), u2s[s - 1]);
		*/
    }

    //Mat uxy[] = {u1s[0], u2s[0]};
    //merge(uxy, 2, _flow);
}