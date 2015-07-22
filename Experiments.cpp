#include "Experiments.h"

void GA_test() {
	string test_set = "res256";
	int n = 4;

	vector<Mat> imgsC1;
	vector<Mat> imgsC3;
	vector<Mat> flows;
	vector<Mat> flows_back;
	vector<Mat> confs;

	imgsC1.resize(n);
	imgsC3.resize(n);
	flows.resize(n);
	flows_back.resize(n);
	confs.resize(n);

	int sigma = 0;
	cout << "read in images\n";
	for (int k = 0; k < n; k++) {
		imgsC1[k] = imread("input/" + test_set + "_0" + int2str(k+1) + ".bmp", CV_LOAD_IMAGE_GRAYSCALE);
		imgsC3[k] = imread("input/" + test_set + "_0" + int2str(k+1) + ".bmp", CV_LOAD_IMAGE_COLOR);

		Rect smaller(0, 0, 20, 20);
		imgsC1[k] = imgsC1[k](smaller);
		imgsC3[k] = imgsC3[k](smaller);
	}

	cout << "calculating flows & confidences\n";
	Mod_OpticalFlowDual_TVL1* OptFlow = new Mod_OpticalFlowDual_TVL1;

	for (int k = 0; k < n; k++) {
		OptFlow->calc(imgsC1[k], imgsC1[0], flows[k]);
		OptFlow->calc(imgsC1[0], imgsC1[k], flows_back[k]);
		showConfidence (flows[k], flows[k], confs[k]);
	}

	delete OptFlow;

	double scale = 2;
	Mat dot = Mat::zeros(5,5,CV_64F);
	dot.at<double>(2, 2) = 1; 

	double PSF_sigma = 0.7 * SQR(scale/2);
	Mat PSF = Mat::zeros(5,5,CV_64F);
	GaussianBlur(dot, PSF, Size( 5, 5), PSF_sigma, PSF_sigma, BORDER_REPLICATE);

	Mat BPk = PSF;

	Mat HRimg, HRimgC3;

	
	GA_Constructor ga_constructor(imgsC1, flows, confs, scale, PSF);
	ga_constructor.solve();
	ga_constructor.output(HRimg);
	//*/

	imwrite("output/" + test_set + "_GA_Construct_HRC1_" + int2str(n) + ".bmp", HRimg);
	outputHRcolor(HRimg, imgsC3[0], HRimgC3);
	imwrite("output/" + test_set + "_GA_Construct_HRC3_" + int2str(n) + ".bmp", HRimgC3);
	/**/
}

void symmetricOptFlow_test() {
	String test_set = "shake2000";	
	int n = 4;

	vector<Mat> imgsC1;
	vector<Mat> flows;
	vector<Mat> flows_back;
	vector<Mat> confs;

	imgsC1.resize(n);
	flows.resize(n);
	flows_back.resize(n);
	confs.resize(n);

	cout << "read in images\n";
	for (int k = 0; k < n; k++) {
		imgsC1[k] = imread("input/" + test_set + "_0" + int2str(k+1) + ".bmp", CV_LOAD_IMAGE_GRAYSCALE);
	}

	for (int k = 0; k < n; k++) {
		cout << "calculating symmetric flows " + int2str(k) + "\n";

		SymmConfOptFlow_calc symmOptFlow;
		symmOptFlow.calc_tv1(imgsC1[k], imgsC1[0], flows[k], flows_back[k], confs[k]);

		imwrite("output/symmConf_" + test_set + int2str(k) + "to0.bmp", confs[k]*255);
	}

	Mat dot = Mat::zeros(5,5,CV_64F);
	dot.at<double>(2, 2) = 1; 

	Mat PSF = Mat::zeros(5,5,CV_64F);
	GaussianBlur(dot, PSF, Size( 5, 5), 0.7, 0.7, BORDER_REPLICATE);

	Mat BPk = PSF;

	Mat HRimg;
	/*
	TermCriteria BPstop;
	BPstop.type = TermCriteria::COUNT + TermCriteria::EPS;
	BPstop.maxCount = 10;
	BPstop.epsilon = 1;

	vector<Mat> bpimg, bpflows;
	bpimg.push_back(imgsC1[3]);
	bpflows.push_back(combineFlows[3]);

	BackProjection_Confidence(HRimg, 2, bpimg, bpflows, PSF, BPk, BPstop, confs);
	*/
	//DivideToBlocksToConstruct( imgsC1, flows, confs, PSF, 2, HRimg);
	/*
	LinearConstructor linearConstructor( imgsC1, combineFlows, combineConfs, 2, PSF);
	linearConstructor.addRegularization_grad2norm(0.05);
	linearConstructor.solve_by_CG();
	linearConstructor.output(HRimg);
	/**/
	
	//imwrite("output/" + test_set + "_LinearConstruct_HR" + int2str(n) + "_SymmConf.bmp", HRimg);
	
	/*writeImgDiff(imread("output/" + test_set + "_LinearConstruct_HR" + int2str(n) + "_CG.bmp", CV_LOAD_IMAGE_GRAYSCALE),
		imread("Origin/" + test_set + "Ori_01.bmp", CV_LOAD_IMAGE_GRAYSCALE),
		"output/" + test_set + "_OriginLinearConstruct" + int2str(n) + "_Diff.bmp");*/

	return;
}

void exampleBased_test ()
{
	String test_set = "res256";

	Mat LR_img_tmp = imread("input/" + test_set + "_01.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	Mat LR_img;
	LR_img_tmp.convertTo(LR_img, CV_64F);

	double scale = 2;
	int LR_rows = LR_img.rows, LR_cols = LR_img.cols;
	int HR_rows = LR_rows * scale, HR_cols = LR_cols * scale;
	int Reconstruct_w = 6, Reconstruct_h = 6;

	Mat HR_img_tmp;
	//resize(LR_img, HR_img, Size(HR_rows, HR_cols));
	//----BP
	Mat HR_BP = imread("input/HR_BP.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	/*
	vector<Mat> imgs, flows;
	imgs.push_back(LR_img);
	Mat noFlow = Mat::zeros(LR_rows, LR_cols, CV_64FC2);
	flows.push_back(noFlow);

	Mat dot = Mat::zeros(5,5,CV_64F);
	dot.at<double>(2, 2) = 1; 
	Mat PSF = Mat::zeros(5,5,CV_64F);
	GaussianBlur(dot, PSF, Size( 5, 5), 1, 1, BORDER_REPLICATE);
	Mat BPk = PSF;
	TermCriteria BPstop;
	BPstop.type = TermCriteria::COUNT + TermCriteria::EPS;
	BPstop.maxCount = 20;
	BPstop.epsilon = 0.1;

	BackProjection ( HR_BP, scale, imgs, flows, PSF, BPk, BPstop );
	imwrite("output/HR_BP.bmp", HR_BP);
	*/
	HR_BP.copyTo(HR_img_tmp);
	
	//----BP
	Mat HR_preset = imread("input/res256_LinearConstruct_HR4_CG_blurGaussian05.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	Mat HR_doneMask = Mat::zeros(HR_rows, HR_cols, CV_8U);

	int preset_x = 128, preset_y = 128;
	int preset_h = 256, preset_w = 256;
	Rect presetRegion= Rect(preset_x, preset_y, preset_w, preset_h);
	HR_preset(presetRegion).copyTo(HR_img_tmp(presetRegion));
	HR_doneMask(presetRegion) = HR_doneMask(presetRegion) + 1;
	Mat HR_img;
	HR_img_tmp.convertTo(HR_img, CV_64F);
	imwrite("output/HR_preset.bmp", HR_img);

	Mat HR_doingMask = Mat::zeros(HR_rows, HR_cols, CV_8U);
	
	bool constructComplete = false;
	while (!constructComplete) {
		constructComplete = true;

		for (int i = Reconstruct_h; i < HR_rows-Reconstruct_h; i+=Reconstruct_h) for (int j = Reconstruct_w; j < HR_cols-Reconstruct_w; j+=Reconstruct_w) {
			if (HR_doneMask.at<uchar>(i, j) != 0)
				continue;

			constructComplete = false;
			int HR_exist[4] = {0};
			bool anyHRnear = false;
			if (HR_doneMask.at<uchar>(i-Reconstruct_h, j) != 0) { // up
				HR_exist[0] = 1;
				anyHRnear = true;
			}
			if (HR_doneMask.at<uchar>(i+Reconstruct_h, j) != 0) { // down
				HR_exist[1] = 1;
				anyHRnear = true;
			}
			if (HR_doneMask.at<uchar>(i, j-Reconstruct_w) != 0) { // left
				HR_exist[2] = 1;
				anyHRnear = true;
			}
			if (HR_doneMask.at<uchar>(i, j+Reconstruct_w) != 0) { // right
				HR_exist[3] = 1;
				anyHRnear = true;
			}

			if (anyHRnear) {
				cout << "fix" << endl;
				Rect currentRect = Rect(j, i, Reconstruct_w, Reconstruct_h);
				cout << i << ", " <<  j << endl;
				cout << HR_exist[0] <<  HR_exist[1] <<HR_exist[2] <<HR_exist[3] <<endl;
				HR_doingMask(currentRect) = 1;
				imwrite("output/doneMask.bmp", HR_doneMask*128);
				imwrite("output/doingMask.bmp", HR_doingMask*128);
				Mat selectPatch;
				mySearch(selectPatch, LR_img, HR_img, currentRect, HR_exist, 1);
				cout << "fix-done" << endl;
				selectPatch.copyTo(HR_img(currentRect));
				imwrite("output/HR_PROGRESS.bmp", HR_img);
			}		

		}

		HR_doneMask = HR_doneMask + HR_doingMask;
		HR_doingMask = Mat::zeros(HR_rows, HR_cols, CV_8U);
	}
	imwrite("output/HR_constructed.bmp", HR_img);
	/**/
}

void test()
{
	String test_set = "res256";
	int n = 4;

	vector<Mat> imgsC1;
	vector<Mat> flows;
	vector<Mat> flows_back;
	vector<Mat> confs;
	
	vector<Mat> blurImg;
	vector<Mat> blur_flows;
	vector<Mat> blur_flows_back;
	vector<Mat> blur_confs;

	vector<Mat> diffs;

	imgsC1.resize(n);
	flows.resize(n);
	flows_back.resize(n);
	confs.resize(n);

	blurImg.resize(n);
	blur_flows.resize(n);
	blur_flows_back.resize(n);
	blur_confs.resize(n);

	diffs.resize(n);

	cout << "read in images\n";
	for (int k = 0; k < n; k++) {
		imgsC1[k] = imread("input/" + test_set + "_0" + int2str(k+1) + ".bmp", CV_LOAD_IMAGE_GRAYSCALE);
		GaussianBlur( imgsC1[k], blurImg[k], Size( 21, 21), 1, 1 );
		imwrite("output/" + test_set + int2str(k) + "_blur.bmp", blurImg[k]);
	}

	cout << "calculating flows & confidences\n";
	Ptr<DenseOpticalFlow> OptFlow = createOptFlow_DualTVL1();	
	for (int k = 0; k < n; k++) {
		cout << "\t calculating " << int2str(k) << " ...\n";

		//flows[k] = Mat::zeros(imgsC1[0].rows, imgsC1[0].cols, CV_64FC2);
		//flows_back[k] = Mat::zeros(imgsC1[0].rows, imgsC1[0].cols, CV_64FC2);
		OptFlow->calc(imgsC1[k], imgsC1[0], flows[k]);
		OptFlow->calc(imgsC1[0], imgsC1[k], flows_back[k]);
		cout << "\t showing Confidence " << int2str(k) << " ...\n";
		showConfidence (flows[k], flows_back[k], confs[k]);

		imwrite("output/conf_" + test_set + int2str(k) + "to0.bmp", confs[k]*254);
		//imwrite("output/confnew_" + test_set + int2str(k) + "to0.bmp", newConfs[k]*254);

		//blur_flows[k] = Mat::zeros(imgsC1[0].rows, imgsC1[0].cols, CV_64FC2);
		//blur_flows_back[k] = Mat::zeros(imgsC1[0].rows, imgsC1[0].cols, CV_64FC2);
		OptFlow->calc(blurImg[k], blurImg[0], blur_flows[k]);
		OptFlow->calc(blurImg[0], blurImg[k], blur_flows_back[k]);
		showConfidence (blur_flows[k], blur_flows_back[k], blur_confs[k]);

		imwrite("output/blurConf_" + test_set + int2str(k) + "to0.bmp", blur_confs[k]*254);

		diffs[k] = Mat::zeros(flows[k].rows, flows[k].cols, CV_64F);
		for (int i = 0; i < flows[k].rows; i++) for (int j = 0; j < flows[k].cols; j++) {
			Vec2f& tmp1 = flows[k].at<Vec2f>(i, j);
			Vec2f& tmp2 = blur_flows[k].at<Vec2f>(i, j);

			diffs[k].at<double>(i, j) =  SQR(tmp1[0] - tmp2[0]) + SQR(tmp1[1] - tmp2[1]);
		}

		imwrite("output/diff_" + test_set + int2str(k) + ".bmp", diffs[k]);
	}



}

void flow2H_test () {
	String test_set = "rubber";
	int n = 4;

	vector<Mat> imgsC1;
	vector<Mat> flows;
	vector<Mat> flows_back;
	vector<Mat> confs;
	vector<Mat> newConfs;

	imgsC1.resize(n);
	flows.resize(n);
	flows_back.resize(n);
	confs.resize(n);
	newConfs.resize(n);

	cout << "read in images\n";
	for (int k = 0; k < n; k++) {
		imgsC1[k] = imread("input/" + test_set + "_0" + int2str(k+1) + ".bmp", CV_LOAD_IMAGE_GRAYSCALE);
	}

	int minHessian = 400;
	SurfFeatureDetector detector( minHessian );
	vector< vector<KeyPoint> >  keypoints;
	keypoints.resize(n);
	for (int k = 0; k < n; k++) {
		detector.detect( imgsC1[k], keypoints[k] );
	}
	SurfDescriptorExtractor extractor;
	vector<Mat> descriptors;
	descriptors.resize(n);
	for (int k = 0; k < n; k++) {
		extractor.compute( imgsC1[k], keypoints[k], descriptors[k] );
	}
	FlannBasedMatcher matcher;
	vector< vector< DMatch > > matches;
	matches.resize(n);
	for (int k = 1; k < n; k++) {
		matcher.match( descriptors[0], descriptors[k], matches[k] );
	}
	

  double max_dist = 0; double min_dist = 100;

	/*
	cout << "calculating flows & confidences\n";
	Ptr<DenseOpticalFlow> OptFlow = createOptFlow_DualTVL1();	
	for (int k = 0; k < n; k++) {
		cout << "\t calculating " << int2str(k) << " ...\n";

		flows[k] = Mat::zeros(imgsC1[0].rows, imgsC1[0].cols, CV_64FC2);
		flows_back[k] = Mat::zeros(imgsC1[0].rows, imgsC1[0].cols, CV_64FC2);
		OptFlow->calc(imgsC1[k], imgsC1[0], flows[k]);
		OptFlow->calc(imgsC1[0], imgsC1[k], flows_back[k]);
		cout << "\t showing Confidence " << int2str(k) << " ...\n";
		showConfidence (flows[k], flows_back[k], confs[k]);
		//cout << "\t showing newConfidence " << int2str(k) << " ...\n";
		//showConfidence_new (flows[k], flows_back[k], newConfs[k]);

		imwrite("output/conf_" + test_set + int2str(k) + "to0.bmp", confs[k]*254);
		//imwrite("output/confnew_" + test_set + int2str(k) + "to0.bmp", newConfs[k]*254);

		newConfs[k] = confs[k];

	}

	double confThresh = 0.1;

	for (int k = 1; k < n; k++) {
		
		vector<Point2f> srcPoints, dstPoints;
		srcPoints.reserve(newConfs[k].rows * newConfs[k].cols);
		dstPoints.reserve(newConfs[k].rows * newConfs[k].cols);

		for (int i = 0; i < newConfs[k].rows; i++) for (int j = 0; j < newConfs[k].cols; j++) {
			if (newConfs[k].at<double>(i, j) > confThresh) {
				Vec2f& tmp_flow = flows[k].at<Vec2f>(i, j);

				srcPoints.push_back(Point2f(j, i));
				dstPoints.push_back(Point2f(j + tmp_flow[0], i + tmp_flow[1]));
			}
		}		

		Mat H, mask, maskImg;

		double cur_err[6] = {0.1, 0.5, 1, 2.5, 5, 10};
		int test_num = 0;

		while (test_num < 6) {

			H = findHomography(srcPoints, dstPoints, CV_RANSAC, cur_err[test_num], mask);

			maskImg = Mat::zeros(newConfs[k].rows, newConfs[k].cols, CV_64F);

			cout << mask.type();

			int count = 0;
			for (int i = 0; i < mask.rows; i++) {
				maskImg.at<double>(srcPoints[i]) = mask.at<short>(i, 1);
				if (mask.at<short>(i, 1) == 0) {
					count++;
				}
			}

			imwrite("output/" + test_set + "_mask" + int2str(test_num) + "_" + int2str(k) + ".png", maskImg*254);
			cout << count << "/" << mask.rows << endl;

			test_num++;
		}
	}
	*/

}

void LinearConstruct_test () {
	String test_set = "bill256";	
	int n = 4;

	vector<Mat> imgsC1;
	vector<Mat> imgsC3;
	vector<Mat> flows;
	vector<Mat> flows_back;
	vector<Mat> confs;

	vector<Mat> preProsImgs;
	vector<Mat> newFlows;
	vector<Mat> newFlows_back;
	vector<Mat> newConfs;	

	vector<Mat> blurImg;
	vector<Mat> blur_flows;
	vector<Mat> blur_flows_back;
	vector<Mat> blur_confs;

	vector<Mat> symm_flows;
	vector<Mat> symm_flows_back;
	vector<Mat> symm_confs;

	vector<Mat> combineFlows;
	vector<Mat> combineConfs;
	vector<Mat> combineFlows2;
	vector<Mat> combineConfs2;

	imgsC1.resize(n);
	imgsC3.resize(n);
	flows.resize(n);
	flows_back.resize(n);
	confs.resize(n);

	preProsImgs.resize(n);
	newFlows.resize(n);
	newFlows_back.resize(n);
	newConfs.resize(n);	

	blurImg.resize(n);
	blur_flows.resize(n);
	blur_flows_back.resize(n);
	blur_confs.resize(n);

	combineFlows.resize(n);
	combineConfs.resize(n);
	combineFlows2.resize(n);
	combineConfs2.resize(n);

	symm_flows.resize(n);
	symm_flows_back.resize(n);
	symm_confs.resize(n);

	int sigma = 2;
	cout << "read in images\n";
	Mat tmp[2];
	for (int k = 0; k < n; k++) {
		imgsC1[k] = imread("input/" + test_set + "_0" + int2str(k+1) + ".bmp", CV_LOAD_IMAGE_GRAYSCALE);
		imgsC3[k] = imread("input/" + test_set + "_0" + int2str(k+1) + ".bmp", CV_LOAD_IMAGE_COLOR);
		//GaussianBlur( imgsC1[k], blurImg[k], Size( floor(sigma*2 + 9), floor(sigma*2 + 9)), sigma, sigma );

		//specialBlur(imgsC1[k], preProsImgs[k]);
		/*
		tmp[0] = imgsC1[k];
		for (int i = 0; i < 4; i++) {
			bilateralFilter(tmp[k%2], tmp[(k+1)%2], 0, 6, 4, BORDER_REPLICATE );
		}
		blurImg[k] = tmp[(k+1)%2];
		*/
	}
	//ImgPreProcess(imgsC1, preProsImgs);

	cout << "calculating flows & confidences\n";
	//Ptr<DenseOpticalFlow> OptFlow = createOptFlow_DualTVL1();
	//Mod_OpticalFlowDual_TVL1* OptFlow = new Mod_OpticalFlowDual_TVL1;
	//OptFlow->setBool("useInitialFlow", true);
	SymmConfOptFlow_calc* symmOptFlow = new SymmConfOptFlow_calc;
	time_t time1, time0;

	for (int k = 0; k < n; k++) {
		//imwrite("output/ImgPre_" + test_set + int2str(k) + ".bmp", preProsImgs[k]);
		//imwrite("output/BlurGaussian" + int2str(sigma) + "_" + test_set + int2str(k) + ".bmp", blurImg[k]);

		time(&time0);

		//optFlowHS(imgsC1[k], imgsC1[0], flows[k]);
		//optFlowHS(imgsC1[0], imgsC1[k], flows[k]);
		//calcOpticalFlowFarneback(imgsC1[k], imgsC1[0], flows[k], 0.5, 5, 1, 300, 19, 2.3, OPTFLOW_FARNEBACK_GAUSSIAN);
		//calcOpticalFlowFarneback(imgsC1[0], imgsC1[k], flows_back[k], 0.5, 5, 1, 300, 19, 2.3, OPTFLOW_FARNEBACK_GAUSSIAN);
		//calcOpticalFlowSF(imgsC3[k], imgsC3[0], flows[k], 5, 1, 3);
		//calcOpticalFlowSF(imgsC3[0], imgsC3[k], flows_back[k], 5, 1, 3);
		//showConfidence (flows[k], flows_back[k], confs[k]);
		/**/		
		/*
		OptFlow->calc(imgsC1[k], imgsC1[0], flows[k]);
		OptFlow->calc(imgsC1[0], imgsC1[k], flows_back[k]);
		showConfidence (flows[k], flows_back[k], confs[k]);
		/**/
		/*
		OptFlow->calc(preProsImgs[k], preProsImgs[0], newFlows[k]);
		OptFlow->calc(preProsImgs[0], preProsImgs[k], newFlows_back[k]);
		showConfidence (newFlows[k], newFlows_back[k], newConfs[k]);		
		/**/
		
		// if we want to use initial flow, need to pre-allocate
		//blur_flows[k] = Mat::zeros(imgsC1[k].rows, imgsC1[k].cols, CV_64FC2);
		//blur_flows_back[k] = Mat::zeros(imgsC1[k].rows, imgsC1[k].cols, CV_64FC2);
		/*
		OptFlow->calc(blurImg[k], blurImg[0], blur_flows[k]);	
		OptFlow->calc(blurImg[0], blurImg[k], blur_flows_back[k]);
		showConfidence (blur_flows[k], blur_flows_back[k], blur_confs[k]);
		/**/
		
		
		symmOptFlow->calc_tv1(imgsC1[k], imgsC1[0], symm_flows[k], symm_flows_back[k], symm_confs[k]);
		/**/

		/*
		Mat check_flow;
		Mat zero = Mat::zeros(symm_flows[k].size(), CV_32FC2);
		calcVecMatDiff (symm_flows[k], zero, check_flow);
		imwrite("output/ggggg.bmp", check_flow*255);
		//*/

		time(&time1);
		cout << "flow" << int2str(k) << ": " << difftime(time1, time0) << endl;
		time0 = time1;

		//imwrite("output/SFconf_" + test_set + int2str(k) + "to0.bmp", confs[k]*254);
		//imwrite("output/conf_" + test_set + int2str(k) + "to0.bmp", confs[k]*254);
		//imwrite("output/newConf_" + test_set + int2str(k) + "to0.bmp", newConfs[k]*254);
		//imwrite("output/blurConfGaussian" + int2str(sigma) + "_" +  test_set + int2str(k) + "to0.bmp", blur_confs[k]*254);
		imwrite("output/Conf_" + test_set + int2str(k) + "to0.bmp", symm_confs[k]*255);
		
		Mat warpImg;
		/*
		warpImageByFlow(imgsC3[k], flows_back[k], warpImg);
		imwrite("output/warpByFlow_" + test_set + int2str(k) + ".bmp", warpImg);
		/**/
		
		//warpImageByFlow(imgsC3[k], symm_flows_back[k], warpImg);
		//warpImageByFlow(imgsC3[0], symm_flows[k], warpImg);
		//imwrite("output/warpto" + int2str(k) + "_" + test_set + ".bmp", warpImg);
		/**/
	}
	//getBetterFlow(confs, flows, newConfs, newFlows, combineConfs2, combineFlows2);
	//getBetterFlow(newConfs, newFlows, blur_confs, blur_flows, combineConfs, combineFlows);
	//getBetterFlow(confs, flows, blur_confs, blur_flows, combineConfs, combineFlows);

	
	for (int k = 0; k < n; k++) {
		time(&time0);

		//calcOpticalFlowFarneback(imgsC1[k], imgsC1[0], flows[k], 0.5, 5, 3, 500, 5, 1.1, OPTFLOW_FARNEBACK_GAUSSIAN);
		//calcOpticalFlowFarneback(imgsC1[0], imgsC1[k], flows_back[k], 0.5, 5, 3, 500, 5, 1.1, OPTFLOW_FARNEBACK_GAUSSIAN);
		//calcOpticalFlowSF(imgsC1[k], imgsC1[0], flows[k], 5, 3, 10);
		//calcOpticalFlowSF(imgsC1[0], imgsC1[k], flows_back[k], 5, 3, 10);
		/*
		OptFlow->calc(imgsC1[k], imgsC1[0], flows[k]);
		OptFlow->calc(imgsC1[0], imgsC1[k], flows_back[k]);
		showConfidence (flows[k], flows_back[k], confs[k]);
		/**/
		/*
		OptFlow->calc(preProsImgs[k], preProsImgs[0], newFlows[k]);
		OptFlow->calc(preProsImgs[0], preProsImgs[k], newFlows_back[k]);
		showConfidence (newFlows[k], newFlows_back[k], newConfs[k]);		
		/**/
		/*
		OptFlow->calc(blurImg[k], blurImg[0], blur_flows[k]);
		OptFlow->calc(blurImg[0], blurImg[k], blur_flows_back[k]);
		showConfidence (blur_flows[k], blur_flows_back[k], blur_confs[k]);
		/**/

		
		//imwrite("output/conf_" + test_set + int2str(k) + "to0.bmp", confs[k]*254);
		//imwrite("output/newConf_" + test_set + int2str(k) + "to0.bmp", newConfs[k]*254);
		//imwrite("output/blurConf2_" + test_set + int2str(k) + "to0.bmp", blur_confs[k]*254);
		
		time(&time1);
		cout << "flow" << int2str(k) << ": " << difftime(time1, time0) << endl;
		time0 = time1;
	}
	
	for (int k = 0; k < n; k++) {
		//flows[k].resize(0, 0);
		//confs[k].resize(0, 0);
		//newFlows[k].resize(0, 0);
		//newConfs[k].resize(0, 0);
		/*
		combineConfs[k] = confs[k];
		combineFlows[k] = flows[k];
		/**/

		//symm_confs[k] = 1;

		combineConfs[k] = symm_confs[k];
		combineFlows[k] = symm_flows[k];

		/**/
		//imwrite("output/combineConfs_" + test_set + int2str(k) + "to0.bmp", combineConfs[k]*254);	

	}
		
	// start reconstruct
	// release 
	//delete OptFlow;
	delete symmOptFlow;

	double scale = 2;
	Mat dot = Mat::zeros(5,5,CV_64F);
	dot.at<double>(2, 2) = 1; 

	double PSF_sigma = 0.7 * SQR(scale/2);
	int PSF_size =  5 * scale / 2;
	Mat PSF = Mat::zeros(PSF_size,PSF_size,CV_64F);
	GaussianBlur(dot, PSF, Size( PSF_size, PSF_size), PSF_sigma, PSF_sigma, BORDER_REPLICATE);

	Mat BPk = PSF;
	
	Mat HRimg, HRimgC3;
	
	Block_Constructor divided2Blocks( imgsC1, combineFlows, combineConfs, scale, PSF);
	divided2Blocks.output(HRimg);

	//DivideToBlocksToConstruct( imgsC1, combineFlows, combineConfs, PSF, scale, HRimg);
	//DivideToBlocksToConstruct( imgsC1, flows, confs, PSF, scale, HRimg);
	/*
	TermCriteria BPstop;
	BPstop.type = TermCriteria::COUNT + TermCriteria::EPS;
	BPstop.maxCount = 500;
	BPstop.epsilon = 1;
	//*/
	//BP_Constructor BPconstructor(HRimg, scale, imgsC1, combineFlows, PSF, BPk, BPstop, combineConfs);

	//BackProjection(HRimg, scale, imgsC1, combineFlows, PSF, BPk, BPstop);
	//BackProjection_Confidence(HRimg, scale, imgsC1, combineFlows, PSF, BPk, BPstop, combineConfs);
	imwrite("output/" + test_set + "_BlockReconstructC1_wConf.bmp", HRimg);
	outputHRcolor(HRimg, imgsC3[0], HRimgC3);
	imwrite("output/" + test_set + "_BlockReconstructC3_wConf.bmp", HRimgC3);
	//*/
	/*
	LinearConstructor linearConstructor( imgsC1, combineFlows, combineConfs, 2, PSF);
	linearConstructor.addRegularization_grad2norm(0.05);
	linearConstructor.solve_by_CG();
	linearConstructor.output(HRimg);
	/**/
	/*
	imwrite("output/" + test_set + "_LinearConstructC1_wConf.bmp", HRimg);
	outputHRcolor(HRimg, imgsC3[0], HRimgC3);
	imwrite("output/" + test_set + "_LinearConstructC3_wConf.bmp", HRimgC3);
	/*
	for (int k = 0; k < n; k++) {
		symm_confs[k] = 1;
	}
	//*/
	//DivideToBlocksToConstruct( imgsC1, combineFlows, symm_confs, PSF, 2, HRimg);
	/*
	DivideToBlocksToConstruct( imgsC1, blur_flows, blur_confs, PSF, 2, HRimg);
	imwrite("output/" + test_set + "_LinearConstructC1_noConf.bmp", HRimg);
	outputHRcolor(HRimg, imgsC3[0], HRimgC3);
	imwrite("output/" + test_set + "_LinearConstructC3_noConf.bmp", HRimgC3);
	//*/
	/*
	DivideToBlocksToConstruct( imgsC1, symm_flows, symm_confs, PSF, 2, HRimg);
	imwrite("output/" + test_set + "_LinearConstruct_HRC1_" + int2str(n) + "_Gaussian" + int2str(sigma) + "_symm.bmp", HRimg);
	outputHRcolor(HRimg, imgsC3[0], HRimgC3);
	imwrite("output/" + test_set + "_LinearConstruct_HRC3_" + int2str(n) + "_Gaussian" + int2str(sigma) + "_symm.bmp", HRimgC3);
	/**/
	return;
}
/*
void FlexISP_test () {
	
	//Mat in = imread("input/test/CIMG1439.JPG", IMREAD_GRAYSCALE);
	//Mat in_double;
	//in.convertTo(in_double, CV_64F);
	//vector<Mat> y;
	//y.resize(3);
	//for (int k = 0; k < 3; k++) {
	//	y[k] = Mat::zeros(in.rows, in.cols, CV_64F);
	//}
	//vector<Mat> output;
	//output.resize(3);
	//for (int k = 0; k < 3; k++) {
	//	output[k] = Mat::zeros(in.rows, in.cols, CV_64F);
	//}

	//penalty(y, in_double, 1);
	//data_fidelity(in_double, in_double, y, 1);

	String test_set = "bear";	
	int n = 4;

	vector<Mat> imgsC1;
	vector<Mat> flows;
	vector<Mat> flows_back;
	vector<Mat> confs;

	imgsC1.resize(n);
	flows.resize(n);
	flows_back.resize(n);
	confs.resize(n);

	cout << "read in images\n";
	for (int k = 0; k < n; k++) {
		imgsC1[k] = imread("input/" + test_set + "256_0" + int2str(k+1) + ".bmp", CV_LOAD_IMAGE_GRAYSCALE);
	}	

	cout << "calculating flows & confidences\n";
	Ptr<DenseOpticalFlow> OptFlow = createOptFlow_DualTVL1();	
	for (int k = 0; k < n; k++) {
		flows[k] = Mat::zeros(imgsC1[0].rows, imgsC1[0].cols, CV_64FC2);
		flows_back[k] = Mat::zeros(imgsC1[0].rows, imgsC1[0].cols, CV_64FC2);
		OptFlow->calc(imgsC1[k], imgsC1[0], flows[k]);
		OptFlow->calc(imgsC1[0], imgsC1[k], flows_back[k]);
		showConfidence (flows[k], flows_back[k], confs[k]);

		imwrite("output/conf" + int2str(k) + "to0.bmp", confs[k]);
	}

	Mat PSF = Mat::zeros(3,3,CV_64F);
	Mat BPk = PSF;
	
	PSF.at<double>(0,0) = 0.0113;
	PSF.at<double>(0,1) = 0.0838;
	PSF.at<double>(0,2) = 0.0113;
	PSF.at<double>(1,0) = 0.0838;
	PSF.at<double>(1,1) = 0.6193;
	PSF.at<double>(1,2) = 0.0838;
	PSF.at<double>(2,0) = 0.0113;
	PSF.at<double>(2,1) = 0.0838;
	PSF.at<double>(2,2) = 0.0113;

	Mat HRimg;
	FlexISPmain (imgsC1, flows, confs, PSF, BPk, 2, HRimg);
	imwrite("output/" + test_set + "FlexISP_HR.bmp", HRimg);

	return ;
}
*/

void OptFlow_BP_test () {
	String test_set = "bear";
	
	vector<Mat> imgsC1;
	vector<Mat> imgsC3;
	vector<Mat> flows;
	vector<Mat> warps;

	imgsC1.resize(4);
	imgsC3.resize(4);
	for (int k = 0; k < 4; k++) {
		imgsC1[k] = imread("input/" + test_set + "256_0" + int2str(k+1) + ".bmp", CV_LOAD_IMAGE_GRAYSCALE);
		imgsC3[k] = imread("input/" + test_set + "256_0" + int2str(k+1) + ".bmp", CV_LOAD_IMAGE_COLOR);
	}

	Ptr<DenseOpticalFlow> OptFlow = createOptFlow_DualTVL1();
	flows.resize(4);
	for (int k = 0; k < 4; k++) {
		flows[k] = Mat::zeros(imgsC1[0].rows, imgsC1[0].cols, CV_64FC2);
		OptFlow->calc(imgsC1[k], imgsC1[0], flows[k]);
	}
	/*
	warps.resize(4);
	for (int k = 0; k < 4; k++) {
		NaiveForwardNNWarp(imgsC3[k], flows[k], warps[k], 3);
		imwrite("output/" + test_set + "256_" + int2str(k+1) + "warpto1.bmp", warps[k]);
	}
	/**/
	

	double scale = 2;
	Mat dot = Mat::zeros(5,5,CV_64F);
	dot.at<double>(2, 2) = 1; 

	double PSF_sigma = 0.7 * SQR(scale/2);
	Mat PSF = Mat::zeros(5,5,CV_64F);
	GaussianBlur(dot, PSF, Size( 5, 5), PSF_sigma, PSF_sigma, BORDER_REPLICATE);

	Mat BPk = PSF;

	TermCriteria BPstop;
	BPstop.type = TermCriteria::COUNT + TermCriteria::EPS;
	BPstop.maxCount = 10;
	BPstop.epsilon = 1;

	vector<Mat> imgs;

	imgs.push_back(imgsC1[0]);

	Mat HRimg;

	BackProjection(HRimg, 2, imgs, flows, PSF, BPk, BPstop);
	imwrite("output/" + test_set + "_HR_singleBP.bmp", HRimg);

	for (int k = 1; k < 4; k++) {
		imgs.push_back(imgsC1[k]);
	}
	BackProjection(HRimg, 2, imgs, flows, PSF, BPk, BPstop);
	imwrite("output/" + test_set + "_HR_multiBP.bmp", HRimg);

	// Back Projection
	/*
	Mat img1 = imread("input/bear256_01.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	Mat img2 = imread("input/bear256_02.bmp", CV_LOAD_IMAGE_GRAYSCALE);

	Mat flow1 = Mat::zeros( img1.rows, img1.cols, CV_64FC2 );
	Mat flow2 = Mat::zeros( img1.rows, img1.cols, CV_64FC2 );

	Ptr<DenseOpticalFlow> gg = createOptFlow_DualTVL1();
	//gg->set("lambda", 0.1);
	gg->calc(img2, img1, flow2);

	vector<Mat> imgs;
	imgs.push_back(img1);
	imgs.push_back(img2);

	vector<Mat> flows;
	flows.push_back(flow1);
	flows.push_back(flow2);

	Mat HRimg;
	Mat PSF = Mat::zeros(3,3,CV_64F);
	Mat BPk = PSF;
	
	PSF.at<double>(0,0) = 0.0113;
	PSF.at<double>(0,1) = 0.0838;
	PSF.at<double>(0,2) = 0.0113;
	PSF.at<double>(1,0) = 0.0838;
	PSF.at<double>(1,1) = 0.6193;
	PSF.at<double>(1,2) = 0.0838;
	PSF.at<double>(2,0) = 0.0113;
	PSF.at<double>(2,1) = 0.0838;
	PSF.at<double>(2,2) = 0.0113;

	TermCriteria BPstop;
	BPstop.type = TermCriteria::COUNT + TermCriteria::EPS;
	BPstop.maxCount = 21;
	BPstop.epsilon = 0.01;

	BackProjection(HRimg, 2, imgs, flows, PSF, BPk, BPstop);

	imwrite("output/testHR_BP.bmp", HRimg);
	/**/

	return ;
}

void OptFlow_ConfBP_test () {
	// four image BP
	
	String test_set = "res";
	int n = 4;
	
	vector<Mat> imgsC1;
	vector<Mat> imgsC3;
	vector<Mat> flows;
	vector<Mat> flows_back;
	vector<Mat> warps;
	vector<Mat> confs;
	vector<Mat> Gx;
	vector<Mat> Gy;
	vector<Mat> imgsProcessed;
	vector<Mat> tmps;
	vector<Mat> curUsing;

	imgsC1.resize(n);
	imgsC3.resize(n);
	flows.resize(n);
	flows_back.resize(n);
	confs.resize(n);
	Gx.resize(n);
	Gy.resize(n);
	imgsProcessed.resize(n);
	tmps.resize(n);
	warps.resize(n);
	curUsing.resize(n);

	for (int k = 0; k < 4; k++) {
		imgsC1[k] = imread("input/" + test_set + "256_0" + int2str(k+1) + ".bmp", CV_LOAD_IMAGE_GRAYSCALE);
		//imgsC3[k] = imread("input/" + test_set + "256_0" + int2str(k+1) + ".bmp", CV_LOAD_IMAGE_COLOR);
	}
	
	double tmp_max = 0;
	for (int k = 0; k < 4; k++) {
		tmp_max = 0;

		GaussianBlur( imgsC1[k], tmps[k], Size(3,3), 0, 0, BORDER_DEFAULT );
		Sobel( imgsC1[k], Gx[k], CV_64F, 1, 0, 3, 1, 0, BORDER_DEFAULT );
		Sobel( imgsC1[k], Gy[k], CV_64F, 0, 1, 3, 1, 0, BORDER_DEFAULT );

		imgsProcessed[k] = Mat::zeros(imgsC1[0].rows, imgsC1[0].cols, CV_64F);
		for (int i = 0; i < imgsC1[0].rows; i++) for (int j = 0; j < imgsC1[0].cols; j++) {
			imgsProcessed[k].at<double>(i,j) = sqrt( SQR(Gx[k].at<double>(i,j)) + SQR(Gy[k].at<double>(i,j)) ) / ( (double) tmps[k].at<uchar>(i,j) + 1 );
			if (imgsProcessed[k].at<double>(i,j) > tmp_max) {
				tmp_max = imgsProcessed[k].at<double>(i,j);
			}
		}
	}
	for (int k = 0; k < 4; k++) {
		imgsProcessed[k].convertTo(imgsProcessed[k], CV_8U, 255.0 / tmp_max, 0);

		imwrite("output/" + test_set + "256_0" + int2str(k+1) + "_preProcess.bmp", imgsProcessed[k]);
	}
	

	for (int k = 0; k < 4; k++) {
		curUsing[k] = imgsProcessed[k];
	}

	
	// careful calculate which flow
	Ptr<DenseOpticalFlow> OptFlow = createOptFlow_DualTVL1();
	for (int k = 0; k < 4; k++) {
		flows[k] = Mat::zeros(curUsing[0].rows, curUsing[0].cols, CV_64FC2);
		flows_back[k] = Mat::zeros(curUsing[0].rows, curUsing[0].cols, CV_64FC2);
		OptFlow->calc(curUsing[k], curUsing[0], flows[k]);
		OptFlow->calc(curUsing[0], curUsing[k], flows_back[k]);
	}
	
	
	for (int k = 0; k < 4; k++) {
		showConfidence (flows[k], flows_back[k], confs[k]);
		imwrite("output/" + test_set + "256_" + int2str(k+1) + "to1_confidence.bmp", confs[k]*254);
	}
	
	/*
	for (int k = 0; k < 4; k++) {
		NaiveForwardNNWarp(imgsC3[k], flows[k], warps[k], 3);
		imwrite("output/" + test_set + "256_" + int2str(k+1) + "warpto1.bmp", warps[k]);
	}
	/**/
	/*
	Mat HRimg;
	Mat PSF = Mat::zeros(3,3,CV_64F);
	Mat BPk = PSF;
	
	PSF.at<double>(0,0) = 0.0113;
	PSF.at<double>(0,1) = 0.0838;
	PSF.at<double>(0,2) = 0.0113;
	PSF.at<double>(1,0) = 0.0838;
	PSF.at<double>(1,1) = 0.6193;
	PSF.at<double>(1,2) = 0.0838;
	PSF.at<double>(2,0) = 0.0113;
	PSF.at<double>(2,1) = 0.0838;
	PSF.at<double>(2,2) = 0.0113;

	TermCriteria BPstop;
	BPstop.type = TermCriteria::COUNT + TermCriteria::EPS;
	BPstop.maxCount = 200;
	BPstop.epsilon = 0.01;

	vector<Mat> imgs;
	
	for (int k = 0; k < 4; k++) {
		imgs.push_back(imgsC1[k]);
	}
	
	BackProjection_Confidence(HRimg, 2, imgs, flows, PSF, BPk, BPstop, confs);
	imwrite("output/" + test_set + "_HR_multiBP.bmp", HRimg);
	/**/

	//ShowConfidence
	/*
	Mat img1 = imread("input/res256_01.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	Mat img2 = imread("input/res256_04.bmp", CV_LOAD_IMAGE_GRAYSCALE);

	Mat flow1to2 = Mat::zeros( img1.rows, img1.cols, CV_64FC2 );
	Mat flow2to1 = Mat::zeros( img1.rows, img1.cols, CV_64FC2 );

	Ptr<DenseOpticalFlow> gg = createOptFlow_DualTVL1();
	//gg->set("lambda", 0.1);
	gg->calc(img2, img1, flow2to1);
	gg->calc(img1, img2, flow1to2);

	Mat conf1to2, conf2to1;

	showConfidence (flow2to1, flow1to2, conf2to1);
	showConfidence (flow1to2, flow2to1, conf1to2);

	//imwrite("output/conf1to2.bmp", conf1to2);
	imwrite("output/conf2to1.bmp", conf2to1);
	/**/

	return ;
}