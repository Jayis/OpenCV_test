#include "Experiments.h"


// try to solve with GA, a disaster
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

void LinearConstruct_test () {
	String test_set = "bb2Crop2000";	
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

	int sigma = 1;
	cout << "read in images\n";
	Mat tmp[2];
	for (int k = 0; k < n; k++) {
		imgsC1[k] = imread("input/" + test_set + "_0" + int2str(k+1) + ".bmp", CV_LOAD_IMAGE_GRAYSCALE);
		imgsC3[k] = imread("input/" + test_set + "_0" + int2str(k+1) + ".bmp", CV_LOAD_IMAGE_COLOR);
	}

	cout << "calculating flows & confidences\n";

	SymmConfOptFlow_calc* symmOptFlow = new SymmConfOptFlow_calc;
	time_t time1, time0;

	for (int k = 0; k < n; k++) {

		time(&time0);		
		// here you can change different optflow method, all in symmOptFlow class
		symmOptFlow->calc_tv1(imgsC1[k], imgsC1[0], symm_flows[k], symm_flows_back[k], symm_confs[k]);
		/**/

		time(&time1);
		cout << "flow" << int2str(k) << ": " << difftime(time1, time0) << endl;
		time0 = time1;

		imwrite("output/Conf_" + test_set + int2str(k) + "to0.bmp", symm_confs[k]*255);
		
		Mat warpImg;
		
		//warpImageByFlow(imgsC3[k], symm_flows_back[k], warpImg);
		//warpImageByFlow(imgsC3[0], symm_flows[k], warpImg);
		//imwrite("output/warpto" + int2str(k) + "_" + test_set + ".bmp", warpImg);
		/**/
	}
	
	for (int k = 0; k < n; k++) {
		// no meaning, just copy
		combineConfs[k] = symm_confs[k];
		combineFlows[k] = symm_flows[k];
		/**/

		//imwrite("output/combineConfs_" + test_set + int2str(k) + "to0.bmp", combineConfs[k]*254);	
	}
		
	// start reconstruct

	// release MEM
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
	Mat HRimg1, diff, colorDiff;
	
	// here "i" = how many blocks you want to divide per axis

	// for CPU_CG min(i) = 4, 3 will swap out
	// for CPU_CG_multi min(i) = 7
	// for GPU_CG_squareMat min(i) = 4
	// for GPU_CG min(i) = 6
	
	for (int i = 16; i < 17; i++) {
		tmp_blockPerAxis = i;
		Block_Constructor divided2Blocks( imgsC1, combineFlows, combineConfs, scale, PSF);
		divided2Blocks.output(HRimg);
		construct_t[i] = tmp_t;
	}
	//*/

	// naive divide method
	//DivideToBlocksToConstruct( imgsC1, combineFlows, combineConfs, PSF, scale, HRimg);
	//DivideToBlocksToConstruct( imgsC1, flows, confs, PSF, scale, HRimg);

	// Linear Constructor Usage
	/*
	Linear_Constructor linearConstructor( imgsC1, combineFlows, combineConfs, scale, PSF);
	linearConstructor.addRegularization_grad2norm(0.05);
	linearConstructor.solve_by_CG();
	linearConstructor.output(HRimg);
	//*/

	// NN Constructor Usage
	/*
	time_t t0, t1;
	time(&t0);
	NN_Constructor NNConstructor( imgsC1, combineFlows, combineConfs, scale, PSF);
	NNConstructor.solve();
	NNConstructor.output(HRimg1);
	time(&t1);

	cout << difftime(t1, t0) << endl;
	//*/

	// use this to see difference of different construction
	/*
	seeMatDiff(HRimg, HRimg1, diff);
	diff *= 2;
	outputHRcolor(diff, imgsC3[0], HRimgC3);
	applyColorMap(HRimgC3, colorDiff, COLORMAP_JET);

	imwrite("output/colorDiff.bmp", colorDiff);
	//*/

	// BP construtor Usage
	/*
	TermCriteria BPstop;
	BPstop.type = TermCriteria::COUNT + TermCriteria::EPS;
	BPstop.maxCount = 500;
	BPstop.epsilon = 0.01;
	
	BP_Constructor BPconstructor(HRimg, scale, imgsC1, combineFlows, PSF, BPk, BPstop, combineConfs);
	//*/
		
	imwrite("output/" + test_set + "_BlockReconstructC1_wConf.bmp", HRimg);
	outputHRcolor(HRimg, imgsC3[0], HRimgC3);
	imwrite("output/" + test_set + "_BlockReconstructC3_wConf.bmp", HRimgC3);
	//*/

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
