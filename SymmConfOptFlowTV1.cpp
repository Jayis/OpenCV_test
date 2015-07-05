#include "SymmConfOptFlowTV1.h"

SymmConfOptFlow_calc::SymmConfOptFlow_calc()
{
	//OptFlow = createOptFlow_DualTVL1();
	//OptFlow_back = createOptFlow_DualTVL1();
	OptFlow = new Mod_OpticalFlowDual_TVL1;
	OptFlow_back = new Mod_OpticalFlowDual_TVL1;

}

void SymmConfOptFlow_calc::calc_HS(InputArray _I0, InputArray _I1, InputOutputArray _flow, InputOutputArray _flow_back, InputOutputArray _conf)
{
	OptFlow->calc_part1(_I0, _I1, _flow);
	OptFlow_back->calc_part1(_I1, _I0, _flow_back);

	nscales = OptFlow->getInt("nscales");
	confs.resize(nscales);
	confs_back.resize(nscales);
	flows.resize(nscales);
	flows_back.resize(nscales);

	for (s = nscales - 1; s >= 0; --s)
    {
		Mat& tmp_curI0 = OptFlow->getI0SpecScale(s);
		Mat& tmp_curI1 = OptFlow_back->getI0SpecScale(s);

		Mat curI0, curI1;
		tmp_curI0.convertTo(curI0, CV_8UC1);
		tmp_curI1.convertTo(curI1, CV_8UC1);

		if (s != nscales - 1) {
			resize(flows[s+1], flows[s], curI0.size(), 0, 0, CV_INTER_CUBIC);
			optFlowHS(curI0, curI1, flows[s], 1);

			resize(flows_back[s+1], flows_back[s], curI1.size(), 0, 0, CV_INTER_CUBIC);
			optFlowHS(curI1, curI0, flows_back[s], 1);
		}
		else {
			optFlowHS(curI0, curI1, flows[s]);
			optFlowHS(curI1, curI0, flows_back[s]);
		}

		showConfidence(flows[s], flows_back[s], confs[s]);
		showConfidence(flows_back[s], flows[s], confs_back[s]);
	}

	flows[0].copyTo(_flow);
	flows_back[0].copyTo(_flow_back);
	confs[0].copyTo(_conf);
}

void SymmConfOptFlow_calc::calc_tv1(InputArray _I0, InputArray _I1, InputOutputArray _flow, InputOutputArray _flow_back, InputOutputArray _conf)
{
	OptFlow->calc_part1(_I0, _I1, _flow);
	OptFlow_back->calc_part1(_I1, _I0, _flow_back);

	// 
	nscales = OptFlow->getInt("nscales");
	confs.resize(nscales);
	confs_back.resize(nscales);
	flows.resize(nscales);
	flows_back.resize(nscales);
	
	bool seeWhichLayer = false;
	if (seeWhichLayer) {
		fromWhichLayer = Mat::zeros(_I0.size(), CV_8U);
	}

	// pyramidal structure for computing the optical flow
    for (s = nscales - 1; s >= 0; --s)
    {
		cout << "s: " << int2str(s) << endl;

		cout << "procSpecScale" << endl;
        // compute the optical flow at the current scale
		OptFlow->procSpecScale(s);
		OptFlow_back->procSpecScale(s);

		cout << "getFlowSpecScale" << endl;
		Mat flow, flow_back, conf, conf_back;
		OptFlow->getFlowSpecScale(s, flow);
		OptFlow_back->getFlowSpecScale(s, flow_back);
		showConfidence(flow, flow_back, conf);
		showConfidence(flow_back, flow, conf_back);

		cout << "getInterpFlowSpecScale" << endl;
		Mat interp_flow, interp_flow_back, interp_conf, interp_conf_back;
		if (/*0 < s &&*/ s < (nscales - 1)) {
			OptFlow->getInterpFlowSpecScale(s, interp_flow);
			OptFlow_back->getInterpFlowSpecScale(s, interp_flow_back);
			showConfidence(interp_flow, interp_flow_back, interp_conf);
			showConfidence(interp_flow_back, interp_flow, interp_conf_back);

			// see diff between interp_flow and calculated_flow
			/*Mat diff, diff_back;
			calcVecMatDiff(flow, interp_flow, diff);
			calcVecMatDiff(flow_back, interp_flow_back, diff_back);
			imwrite("output/flowDiffbetweenL" + int2str(s) + ".png", diff*255);
			imwrite("output/flowDiffBackbetweenL" + int2str(s) + ".png", diff_back*255);
			/**/

			// see area with high confidence at both layer
			Mat ConsistHighConf = Mat::zeros(conf.size(), CV_64F);
			Mat lastHigher = Mat::zeros(conf.size(), CV_64F);
			Mat thisHigher = Mat::zeros(conf.size(), CV_64F);
			Mat outOf1Pix = Mat::zeros(conf.size(), CV_64F);

			for (int i = 0; i < conf.rows; i++) for (int j = 0; j < conf.cols; j++)
			{
				if (conf.at<double>(i, j) > 0.1 && interp_conf.at<double>(i, j) > 0.1)
				{
					ConsistHighConf.at<double>(i, j) = 1;
				}

				if (conf.at<double>(i, j) >= interp_conf.at<double>(i, j))
				{
					thisHigher.at<double>(i, j) = 1;
				}
				else
				{
					lastHigher.at<double>(i, j) = 1;
				}

				if ( conf.at<double>(i, j) < 0.1 )
				{
					outOf1Pix.at<double>(i, j) = 1;
				}
			}
			imwrite("output/consistHighConf" + int2str(s) + ".png", ConsistHighConf*255);
			imwrite("output/lastHigher" + int2str(s) + ".png", lastHigher*255);
			imwrite("output/thisHigher" + int2str(s) + ".png", thisHigher*255);
			imwrite("output/outOf1Pix" + int2str(s) + ".png", thisHigher*255);

		}
		else {
			interp_conf = -1 * Mat::ones(flow.rows, flow.cols, CV_64F);
			interp_conf_back = -1 * Mat::ones(flow.rows, flow.cols, CV_64F);
		}

		

		// (Harry) before upscale the optical flow, select higher confidence form last layer
		cout << "combineFlow" << endl;

		//selectHigherConf(OptFlow, flow, interp_flow, conf, interp_conf, flows[s], confs[s]);
		//selectHigherConf(OptFlow_back, flow_back, interp_flow_back, conf_back, interp_conf_back, flows_back[s], confs_back[s]);
		
		//fillLowConf_WithH(OptFlow, flow, interp_flow, conf, interp_conf, flows[s], confs[s]);
		//fillLowConf_WithH(OptFlow_back, flow_back, interp_flow_back, conf_back, interp_conf_back, flows_back[s], confs_back[s]);
		
		//fillLowConf_WithLaplaceEQ(OptFlow, flow, interp_flow, conf, interp_conf, flows[s], confs[s]);
		//fillLowConf_WithLaplaceEQ(OptFlow_back, flow_back, interp_flow_back, conf_back, interp_conf_back, flows_back[s], confs_back[s]);
		
		patchI0_whileLowConf(OptFlow, flow, interp_flow, conf, interp_conf, flows[s], confs[s]);
		patchI0_whileLowConf(OptFlow_back, flow_back, interp_flow_back, conf_back, interp_conf_back, flows_back[s], confs_back[s]);
		
		showConfidence(flows[s], flows_back[s], confs[s]);
		showConfidence(flows_back[s], flows[s], confs_back[s]);
	
		// see which layer
		if (seeWhichLayer) {
			Mat fromThisLayer = Mat::zeros(flow.rows, flow.cols, CV_64F);

			for (int i = 0; i < conf.rows; i++) for (int j = 0; j < conf.cols; j++) {
				if (conf.at<double>(i, j) > interp_conf.at<double>(i, j)) {
					fromThisLayer.at<double>(i, j) = 1;
				}
			}

			resize(fromThisLayer, fromThisLayer, fromWhichLayer.size(), 0, 0, INTER_CUBIC);

			for (int i = 0; i < fromThisLayer.rows; i++) for (int j = 0; j < fromThisLayer.cols; j++) {
				if (fromThisLayer.at<double>(i, j) >= 0.5) {
					fromWhichLayer.at<uchar>(i, j) = fromThisLayer.at<double>(i, j) * (255 / nscales) * s;
				}
			}

			imwrite("output/FromLayer" + int2str(s) + ".png", fromThisLayer*255);
		}

		// if this was the last scale, finish now
        if (s == 0) {
			//fillLowConf_WithH(OptFlow, flow, interp_flow, conf, interp_conf, flows[s], confs[s]);
			//fillLowConf_WithH(OptFlow_back, flow_back, interp_flow_back, conf_back, interp_conf_back, flows_back[s], confs_back[s]);
			
			//fillLowConf_WithLaplaceEQ(OptFlow, flow, interp_flow, conf, interp_conf, flows[s], confs[s]);
			//fillLowConf_WithLaplaceEQ(OptFlow_back, flow_back, interp_flow_back, conf_back, interp_conf_back, flows_back[s], confs_back[s]);
			//showConfidence(flows[s], flows_back[s], confs[s]);
			//showConfidence(flows_back[s], flows[s], confs_back[s]);
            /**/
			break;
		}

		cout << "setFlowForNextScale" << endl;
        // otherwise, upsample the optical flow
		OptFlow->setFlowForNextScale(s, flows[s]);
		OptFlow_back->setFlowForNextScale(s, flows_back[s]);
		
    }

	if (seeWhichLayer) {
		imwrite("output/FromWhichLayer.png", fromWhichLayer);
	}

    //Mat uxy[] = {u1s[0], u2s[0]};
    //merge(uxy, 2, _flow);
	flows[0].copyTo(_flow);
	flows_back[0].copyTo(_flow_back);
	confs[0].copyTo(_conf);
}

void SymmConfOptFlow_calc::selectHigherConf(Mod_OpticalFlowDual_TVL1* curOptFlow, Mat& flow, Mat& interp_flow, Mat& conf, Mat& interp_conf, Mat& combined_flow, Mat& combined_conf)
{
	combined_flow = Mat::zeros(flow.size(), CV_32FC2);
	combined_conf = Mat::zeros(flow.size(), CV_64F);

	for (int i = 0; i < conf.rows; i++) for (int j = 0; j < conf.cols; j++) {
		//if (conf.at<double>(i, j) < interp_conf.at<double>(i, j)) {		
			//if (conf.at<double>(i, j) < 0.1 && interp_conf.at<double>(i, j) > 0) {	
		if (false) {
			combined_flow.at<Vec2f>(i, j) = interp_flow.at<Vec2f>(i, j);
			combined_conf.at<double>(i, j) = interp_conf.at<double>(i, j);
		}
		else {
			combined_flow.at<Vec2f>(i, j) = flow.at<Vec2f>(i, j);
			combined_conf.at<double>(i, j) = conf.at<double>(i, j);
		}
	}

}

void SymmConfOptFlow_calc::propagateFlow_WithH(Mod_OpticalFlowDual_TVL1* curOptFlow, Mat& flow, Mat& interp_flow, Mat& conf, Mat& interp_conf, Mat& combined_flow, Mat& combined_conf)
{
	// (Harry) before upscale the optical flow
	// (1) check well aligned pixel (mismatch within 1 pixel range), both layer
	//     add it to flann database
	// (2) for wrong aligned pixel (mismatch over 1 pixel range)
	//     calculate flow for them by calculate H of KNN pixels 

	Mat& curI0 = curOptFlow->getI0SpecScale(s);

	combined_flow = Mat::zeros(flow.size(), CV_32FC2);
	combined_conf = Mat::zeros(flow.size(), CV_64F);

	// building (1)
	double highConf = 0.5;

	vector<Vec3f> highConfPixels;	

	for (int i = 0; i < conf.rows; i++) for (int j = 0; j < conf.cols; j++) {
		if (conf.at<double>(i, j) > highConf /*&& interp_conf.at<double>(i, j) > highConf*/)
		{
			//highConfPixels.push_back( Vec3f( i, j, curI0.at<float>(i, j) ) );
			highConfPixels.push_back( Vec3f( i, j, 0) );
		}
	}

	Mat highConfPixelsMat = Mat(highConfPixels).reshape(1);
	flann::KMeansIndexParams indexParams;
	flann::Index kdtree(highConfPixelsMat, indexParams);

	// query (2)
	int knnK = 30;
	for (int i = 0; i < conf.rows; i++) for (int j = 0; j < conf.cols; j++) {
		if ( conf.at<double>(i, j) < 0.1 && interp_conf.at<double>(i, j) > 0) {
		//if ( conf.at<double>(i, j) < interp_conf.at<double>(i, j) && highConfPixels.size() > knnK) {
			vector<float> query;
			query.push_back(i);
			query.push_back(j);
			//query.push_back(curI0.at<float>(i, j));
			query.push_back(0);

			vector<int> indices(knnK);
			vector<float> dists(knnK);

			//kdtree.radiusSearch(query, indices, dists, 2, 3);
			kdtree.knnSearch(query, indices, dists, knnK);

			vector<Point2f> srcPoints, dstPoints;

			for (int k = 0; k < knnK; k++) {
				int cur_i = highConfPixels[indices[k]].val[0];
				int cur_j = highConfPixels[indices[k]].val[1];

				Vec2f& oldFlow = interp_flow.at<Vec2f>(cur_i ,cur_j);
				Vec2f& newFlow = flow.at<Vec2f>(cur_i ,cur_j);

				srcPoints.push_back(Point2f(cur_j + oldFlow.val[0], cur_i + oldFlow.val[1]));
				dstPoints.push_back(Point2f(cur_j + newFlow.val[0], cur_i + newFlow.val[1]));
			}

			Mat H = findHomography(srcPoints, dstPoints, CV_RANSAC, 1);
			//cout << H.type();
			Mat curP = Mat::zeros(3, 3, CV_64F);
			curP.at<double>(0,0) = j + interp_flow.at<Vec2f>(i ,j).val[0];
			curP.at<double>(1,0) = i + interp_flow.at<Vec2f>(i ,j).val[1];
			curP.at<double>(2,0) = 1;
			Mat newP = H.mul(curP);

			combined_conf.at<double>(i, j) = interp_conf.at<double>(i, j);
			combined_flow.at<Vec2f>(i, j)[0] = newP.at<double>(0,0) - j;
			combined_flow.at<Vec2f>(i, j)[1] = newP.at<double>(1,0) - i;			
		}
		else // conf > 0.1, i.e. within 1 pixel error
		{
			combined_conf.at<double>(i, j) = conf.at<double>(i, j);
			combined_flow.at<Vec2f>(i, j) = flow.at<Vec2f>(i, j);
		}
	}



}

void SymmConfOptFlow_calc::fillLowConf_WithH(Mod_OpticalFlowDual_TVL1* curOptFlow, Mat& flow, Mat& interp_flow, Mat& conf, Mat& interp_conf, Mat& combined_flow, Mat& combined_conf)
{
	Mat& curI0 = curOptFlow->getI0SpecScale(s);

	combined_flow = Mat::zeros(flow.size(), CV_32FC2);
	combined_conf = Mat::zeros(flow.size(), CV_64F);

	// building (1)
	double highConf = 0.5;

	vector<Vec3f> highConfPixels;	

	for (int i = 0; i < conf.rows; i++) for (int j = 0; j < conf.cols; j++) {
		if (conf.at<double>(i, j) > highConf /*&& interp_conf.at<double>(i, j) > highConf*/)
		{
			//highConfPixels.push_back( Vec3f( i, j, curI0.at<float>(i, j) ) );
			highConfPixels.push_back( Vec3f( i, j, 0) );
		}
	}

	Mat highConfPixelsMat = Mat(highConfPixels).reshape(1);
	flann::KMeansIndexParams indexParams;
	flann::Index kdtree(highConfPixelsMat, indexParams);

	// query (2)
	int knnK = 20;
	for (int i = 0; i < conf.rows; i++) for (int j = 0; j < conf.cols; j++) {
		if ( conf.at<double>(i, j) < highConf) {
		//if ( conf.at<double>(i, j) < interp_conf.at<double>(i, j) && highConfPixels.size() > knnK) {
			vector<float> query;
			query.push_back(i);
			query.push_back(j);
			//query.push_back(curI0.at<float>(i, j));
			query.push_back(0);

			vector<int> indices(knnK);
			vector<float> dists(knnK);

			//kdtree.radiusSearch(query, indices, dists, 2, 3);
			kdtree.knnSearch(query, indices, dists, knnK);

			vector<Point2f> srcPoints, dstPoints;

			for (int k = 0; k < knnK; k++) {
				int cur_i = highConfPixels[indices[k]].val[0];
				int cur_j = highConfPixels[indices[k]].val[1];

				Vec2f& Flow = flow.at<Vec2f>(cur_i ,cur_j);

				srcPoints.push_back(Point2f(cur_j, cur_i));
				dstPoints.push_back(Point2f(cur_j + Flow.val[0], cur_i + Flow.val[1]));
			}

			Mat H = findHomography(srcPoints, dstPoints, CV_RANSAC, 1);
			//cout << H.type();
			Mat curP = Mat::zeros(3, 3, CV_64F);
			curP.at<double>(0,0) = j;
			curP.at<double>(1,0) = i;
			curP.at<double>(2,0) = 1;
			Mat newP = H.mul(curP);

			combined_conf.at<double>(i, j) = interp_conf.at<double>(i, j);
			combined_flow.at<Vec2f>(i, j)[0] = newP.at<double>(0,0) - j;
			combined_flow.at<Vec2f>(i, j)[1] = newP.at<double>(1,0) - i;			
		}
		else // conf > 0.1, i.e. within 1 pixel error
		{
			combined_conf.at<double>(i, j) = conf.at<double>(i, j);
			combined_flow.at<Vec2f>(i, j) = flow.at<Vec2f>(i, j);
		}
	}
}

void SymmConfOptFlow_calc::fillLowConf_WithLaplaceEQ(Mod_OpticalFlowDual_TVL1* curOptFlow, Mat& flow, Mat& interp_flow, Mat& conf, Mat& interp_conf, Mat& combined_flow, Mat& combined_conf)
{
	Mat& curI0 = curOptFlow->getI0SpecScale(s);

	combined_flow = Mat::zeros(flow.size(), CV_32FC2);
	combined_conf = Mat::zeros(flow.size(), CV_64F);

	Mat tmp_flow = Mat::zeros(flow.size(), CV_32FC2);

	double err = EX_big;

	while (err > 0.001) {
		err = 0;
		for (int i = 1; i < flow.rows-1; i++) for (int j = 1; j < flow.cols-1; j++)
		{
			// 0.5623 = 0.5 pixel miss
			
			if (conf.at<double>(i, j) < 0.5) {
				/*
				tmp_flow.at<Vec2f>(i, j) = 0.25 * ( 
										flow.at<Vec2f>(i+1, j) +
										flow.at<Vec2f>(i-1, j) +
										flow.at<Vec2f>(i, j+1) +
										flow.at<Vec2f>(i, j-1)										
										);
										//*/
				
				float curPixVal = curI0.at<float>(i, j);
				double w10 = ExpNegSQR(curI0.at<float>(i+1, j), curPixVal, 10), w_10 = ExpNegSQR(curI0.at<float>(i-1, j), curPixVal, 10),
					w01 = ExpNegSQR(curI0.at<float>(i, j+1), curPixVal, 10), w0_1 = ExpNegSQR(curI0.at<float>(i, j-1), curPixVal, 10);
				if ((w10 + w_10 + w01 + w0_1) < EX_small) {
					tmp_flow.at<Vec2f>(i, j) = 0.25 * ( 
										flow.at<Vec2f>(i+1, j) +
										flow.at<Vec2f>(i-1, j) +
										flow.at<Vec2f>(i, j+1) +
										flow.at<Vec2f>(i, j-1)										
										);
				}
				else {
					tmp_flow.at<Vec2f>(i, j) = ( 
										w10*flow.at<Vec2f>(i+1, j) +
										w_10*flow.at<Vec2f>(i-1, j) +
										w01*flow.at<Vec2f>(i, j+1) +
										w0_1*flow.at<Vec2f>(i, j-1)										
										) 
										/ 
										(w10 + w_10 + w01 + w0_1);
										//*/
				}

				Vec2f diff = tmp_flow.at<Vec2f>(i, j) - flow.at<Vec2f>(i, j);

				err += (SQR(diff.val[0]) + SQR(diff.val[1]));
			}
			else {
				tmp_flow.at<Vec2f>(i, j) = flow.at<Vec2f>(i, j);
			}
		}

		for (int i = 1; i < flow.rows-1; i++) for (int j = 1; j < flow.cols-1; j++)
		{
			flow.at<Vec2f>(i, j) = tmp_flow.at<Vec2f>(i, j);
		}
	}

	for (int i = 0; i < flow.rows; i++) for (int j = 0; j < flow.cols; j++)
	{
		combined_flow.at<Vec2f>(i, j) = flow.at<Vec2f>(i, j);
		combined_conf.at<double>(i, j) = conf.at<double>(i, j);
	}
}

void SymmConfOptFlow_calc::patchI0_whileLowConf(Mod_OpticalFlowDual_TVL1* curOptFlow, Mat& flow, Mat& interp_flow, Mat& conf, Mat& interp_conf, Mat& combined_flow, Mat& combined_conf)
{


	combined_flow = Mat::zeros(flow.size(), CV_32FC2);
	combined_conf = Mat::zeros(flow.size(), CV_64F);
	if (s < nscales - 1) {
		Mat& curI0 = curOptFlow->getI0SpecScale(s);
		imwrite("output/" + int2str(s) + "_original.bmp", curI0);
		Mat& lastI0 = curOptFlow->getI0SpecScale(s+1);
		Mat interpLastI0;
		resize(lastI0, interpLastI0, curI0.size(), 0.0, 0.0, CV_INTER_CUBIC);
		if (conf.rows != curI0.rows) {
			cout << "nooo, wrong size\n";
		}
		for (int i = 0; i < conf.rows; i++) for (int j = 0; j < conf.cols; j++) {
			//if (conf.at<double>(i, j) < interp_conf.at<double>(i, j)) {		
			//if (conf.at<double>(i, j) < 0.5 && interp_conf.at<double>(i, j) > 0) {
			if (conf.at<double>(i, j) < interp_conf.at<double>(i, j) && conf.at<double>(i, j) < 0.7) {		
			//if (false) {
				//combined_flow.at<Vec2f>(i, j) = interp_flow.at<Vec2f>(i, j);
				//combined_conf.at<double>(i, j) = interp_conf.at<double>(i, j);
				curI0.at<float>(i, j) = interpLastI0.at<float>(i, j);
			}
			else {
				//combined_flow.at<Vec2f>(i, j) = flow.at<Vec2f>(i, j);
				//combined_conf.at<double>(i, j) = conf.at<double>(i, j);
			}
		}
		imwrite("output/" + int2str(s) + "_modified_higher05.bmp", curI0);
	}

	curOptFlow->procSpecScale(s);
	Mat tmp_flow;
	curOptFlow->getFlowSpecScale(s, tmp_flow);
	for (int i = 0; i < combined_flow.rows; ++i) for (int j = 0; j < combined_flow.cols; j++) {
		combined_flow.at<Vec2f>(i, j) = tmp_flow.at<Vec2f>(i, j);
	}

}

SymmConfOptFlow_calc::~SymmConfOptFlow_calc()
{
	delete OptFlow;
	delete OptFlow_back;
}