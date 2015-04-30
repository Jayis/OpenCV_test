#include "Experiments.h"

void flow2H_test () {
	String test_set = "shop2000";
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

	cout << "calculating flows & confidences\n";
	Ptr<DenseOpticalFlow> OptFlow = createOptFlow_DualTVL1();	
	for (int k = 0; k < n; k++) {
		cout << "\t calculating " << int2str(k) << " ...\n";

		flows[k] = Mat::zeros(imgsC1[0].rows, imgsC1[0].cols, CV_32FC2);
		flows_back[k] = Mat::zeros(imgsC1[0].rows, imgsC1[0].cols, CV_32FC2);
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

		float cur_err[6] = {0.1, 0.5, 1, 2.5, 5, 10};
		int test_num = 0;

		while (test_num < 6) {

			H = findHomography(srcPoints, dstPoints, CV_RANSAC, cur_err[test_num], mask);

			maskImg = Mat::zeros(newConfs[k].rows, newConfs[k].cols, CV_32F);

			cout << mask.type();

			int count = 0;
			for (int i = 0; i < mask.rows; i++) {
				maskImg.at<float>(srcPoints[i]) = mask.at<short>(i, 1);
				if (mask.at<short>(i, 1) == 0) {
					count++;
				}
			}

			imwrite("output/" + test_set + "_mask" + int2str(test_num) + "_" + int2str(k) + ".png", maskImg*254);
			cout << count << "/" << mask.rows << endl;

			test_num++;
		}
	}


}

void LinearConstruct_test () {
	String test_set = "res2000";	
	int n = 4;

	vector<Mat> imgsC1;
	vector<Mat> flows;
	vector<Mat> flows_back;
	vector<Mat> confs;
	vector<Mat> preProsImgs;
	vector<Mat> newFlows;
	vector<Mat> newFlows_back;
	vector<Mat> newConfs;
	vector<Mat> combineFlows;
	vector<Mat> combineConfs;

	imgsC1.resize(n);
	flows.resize(n);
	flows_back.resize(n);
	confs.resize(n);
	preProsImgs.resize(n);
	newFlows.resize(n);
	newFlows_back.resize(n);
	newConfs.resize(n);
	combineFlows.resize(n);
	combineConfs.resize(n);

	cout << "read in images\n";
	for (int k = 0; k < n; k++) {
		imgsC1[k] = imread("input/" + test_set + "_0" + int2str(k+1) + ".bmp", CV_LOAD_IMAGE_GRAYSCALE);		
	}
	ImgPreProcess(imgsC1, preProsImgs);

	cout << "calculating flows & confidences\n";
	Ptr<DenseOpticalFlow> OptFlow = createOptFlow_DualTVL1();	
	for (int k = 0; k < n; k++) {
		flows[k] = Mat::zeros(imgsC1[0].rows, imgsC1[0].cols, CV_32FC2);
		flows_back[k] = Mat::zeros(imgsC1[0].rows, imgsC1[0].cols, CV_32FC2);
		OptFlow->calc(imgsC1[k], imgsC1[0], flows[k]);
		OptFlow->calc(imgsC1[0], imgsC1[k], flows_back[k]);
		showConfidence (flows[k], flows_back[k], confs[k]);

		newFlows[k] = Mat::zeros(imgsC1[0].rows, imgsC1[0].cols, CV_32FC2);
		newFlows_back[k] = Mat::zeros(imgsC1[0].rows, imgsC1[0].cols, CV_32FC2);
		OptFlow->calc(preProsImgs[k], preProsImgs[0], newFlows[k]);
		OptFlow->calc(preProsImgs[0], preProsImgs[k], newFlows_back[k]);
		showConfidence (newFlows[k], newFlows_back[k], newConfs[k]);		

		imwrite("output/conf_" + test_set + int2str(k) + "to0.bmp", confs[k]*254);
		imwrite("output/newConf_" + test_set + int2str(k) + "to0.bmp", newConfs[k]*254);

	}
	getBetterFlow(confs, flows, newConfs, newFlows, combineConfs, combineFlows);
	for (int k = 0; k < n; k++) {
		flows[k].resize(0, 0);
		confs[k].resize(0, 0);
		newFlows[k].resize(0, 0);
		newConfs[k].resize(0, 0);

		imwrite("output/combineConfs_" + test_set + int2str(k) + "to0.bmp", combineConfs[k]*254);
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
	DivideToBlocksToConstruct( imgsC1, combineFlows, combineConfs, PSF, 2, HRimg);
	/*
	LinearConstructor linearConstructor( imgsC1, combineFlows, combineConfs, 2, PSF);
	linearConstructor.addRegularization_grad2norm(0.05);
	linearConstructor.solve_byCG();
	linearConstructor.output(HRimg);
	/**/
	
	imwrite("output/" + test_set + "_LinearConstruct_HR" + int2str(n) + "_CG.bmp", HRimg);
	
	writeImgDiff(imread("output/" + test_set + "_LinearConstruct_HR" + int2str(n) + "_CG.bmp", CV_LOAD_IMAGE_GRAYSCALE),
		imread("Origin/" + test_set + "Ori_01.bmp", CV_LOAD_IMAGE_GRAYSCALE),
		"output/" + test_set + "_OriginLinearConstruct" + int2str(n) + "_Diff.bmp");

	return;
}

void FlexISP_test () {
	/*
	Mat in = imread("input/test/CIMG1439.JPG", IMREAD_GRAYSCALE);
	Mat in_double;
	in.convertTo(in_double, CV_64F);
	vector<Mat> y;
	y.resize(3);
	for (int k = 0; k < 3; k++) {
		y[k] = Mat::zeros(in.rows, in.cols, CV_64F);
	}
	vector<Mat> output;
	output.resize(3);
	for (int k = 0; k < 3; k++) {
		output[k] = Mat::zeros(in.rows, in.cols, CV_64F);
	}

	penalty(y, in_double, 1);
	data_fidelity(in_double, in_double, y, 1);
	*/	

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
		flows[k] = Mat::zeros(imgsC1[0].rows, imgsC1[0].cols, CV_32FC2);
		flows_back[k] = Mat::zeros(imgsC1[0].rows, imgsC1[0].cols, CV_32FC2);
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
		flows[k] = Mat::zeros(imgsC1[0].rows, imgsC1[0].cols, CV_32FC2);
		OptFlow->calc(imgsC1[k], imgsC1[0], flows[k]);
	}
	/*
	warps.resize(4);
	for (int k = 0; k < 4; k++) {
		NaiveForwardNNWarp(imgsC3[k], flows[k], warps[k], 3);
		imwrite("output/" + test_set + "256_" + int2str(k+1) + "warpto1.bmp", warps[k]);
	}
	/**/
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
	BPstop.maxCount = 10;
	BPstop.epsilon = 1;

	vector<Mat> imgs;

	imgs.push_back(imgsC1[0]);

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

	Mat flow1 = Mat::zeros( img1.rows, img1.cols, CV_32FC2 );
	Mat flow2 = Mat::zeros( img1.rows, img1.cols, CV_32FC2 );

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
		flows[k] = Mat::zeros(curUsing[0].rows, curUsing[0].cols, CV_32FC2);
		flows_back[k] = Mat::zeros(curUsing[0].rows, curUsing[0].cols, CV_32FC2);
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

	Mat flow1to2 = Mat::zeros( img1.rows, img1.cols, CV_32FC2 );
	Mat flow2to1 = Mat::zeros( img1.rows, img1.cols, CV_32FC2 );

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