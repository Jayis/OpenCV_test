#include "BackProjection.h"

void BackProjection ( Mat& HRimg, double scale, vector<Mat>& imgs, vector<Mat>& flows, Mat& PSF, Mat& BPk, TermCriteria& BPstop )
{
	// all LR images should have same Size()

	// parameter
	double BP_c = 1;
	double interp_scale = 1000;

	int i, j, k;

	int LR_rows = imgs[0].rows;
	int LR_cols = imgs[0].cols;
	int HR_rows = LR_rows * scale;
	int HR_cols = LR_cols * scale;
	int LR_imgCount = imgs.size();

	// pre-interpolation
	Mat super_PSF;
	Mat super_BPk;	
	preInterpolation ( PSF, super_PSF, interp_scale);
	preInterpolation ( BPk, super_BPk, interp_scale);

	//imwrite("output/PSF.bmp", super_PSF*255);

	//----- for every pixel x of HR image, record influenced pixel y
	// initialize bucket
	HR_Pixel_Array* HR_pixels = new HR_Pixel_Array(HR_rows, HR_cols);
	// initialize influenced pixels (for each pixel in each LR img)
	LR_Pixel_Array* LR_pixels = new LR_Pixel_Array(LR_imgCount, LR_rows, LR_cols);
	//
	/*formInfluenceRelation (imgs,
							flows,
							LR_pixels,
							HR_pixels,
							scale,
							super_PSF,
							super_BPk,
							interp_scale);*/
	InfluenceRelation relations(imgs,
							flows,
							LR_pixels,
							HR_pixels,
							scale,
							super_PSF,
							super_BPk,
							interp_scale);
	

	// test bucket, forming HR initial guess
	HRimg = Mat::zeros(HR_rows, HR_cols, CV_64F);
	double tmp_sum;
	for (i = 0; i < HR_rows; i++) {
		for (j = 0; j < HR_cols; j++) {
			tmp_sum = 0;

			for (k = 0; k < HR_pixels->access(i, j).influence_link_cnt; k++) {
				tmp_sum += relations.influence_links[HR_pixels->access(i, j).influence_link_start + k].pixel->val;
			}
			if (HR_pixels->access(i, j).influence_link_cnt == 0) 
				HRimg.at<double>(i,j) = tmp_sum;
			else
				HRimg.at<double>(i,j) = tmp_sum / HR_pixels->access(i, j).influence_link_cnt;

			//for (k = 0; k < HR_pixels->access(i, j).influenced_pixels.size(); k++) {
			//	tmp_sum += HR_pixels->access(i, j).influenced_pixels[k].pixel->val;
			//}
			//if (HR_pixels->access(i, j).influenced_pixels.size() == 0) 
			//	HRimg.at<double>(i,j) = tmp_sum;
			//else
			//	HRimg.at<double>(i,j) = tmp_sum / HR_pixels->access(i, j).influenced_pixels.size();
		}
	}
	/**/

	double sum_diff, sum_hBP;
	double cur_hBP, diff;

	Mat LR_per;
	// start Back Projection

	int iter = 1;
	double epsi = 0;
	bool stop = false;
	while (!stop) {
		epsi = 0;

		HR_to_LR_percetion(HRimg, *LR_pixels, relations, super_PSF, true, interp_scale);

		// see LR perception
		/*
		LR_per = Mat::zeros(LR_rows, LR_cols, CV_64F);
		for (k = 0; k < imgs.size(); k++) {
			for (i = 0; i < LR_rows; i++) for (j = 0; j < LR_cols; j++) {
				LR_per.at<double>(i,j) = LR_pixels[k][i][j].percetion;
			}
			String LR_p_name = "output/LR_P_" + int2str(k) + "_" + int2str(iter) + ".bmp";
			imwrite(LR_p_name, LR_per);
		}
		/**/


		// for each HR x, calculate f(x)'n+1'
		for (i = 0; i < HR_rows; i++) for (j = 0; j < HR_cols; j++) {
			sum_diff = 0;
			sum_hBP = HR_pixels->access(i, j).hBP_sum;
			// for all influenced pixel
			// sum up hBP first
			/*
			for (k = 0; k < HR_pixels[i][j].influenced_pixels.size(); k++) {
				sum_hBP += HR_pixels[i][j].influenced_pixels[k].hBP;
			}
			*/
			// sum up diff
			for (k = 0; k < HR_pixels->access(i, j).influence_link_cnt; k++) {
				Influenced_Pixel& cur_influenced_pix = relations.influence_links[HR_pixels->access(i, j).influence_link_start + k];
				diff = cur_influenced_pix.pixel->val - cur_influenced_pix.pixel->perception;
				cur_hBP = cur_influenced_pix.hBP;
				sum_diff += diff * ( SQR(cur_hBP) / BP_c / sum_hBP);
			}

			// update HR
			HRimg.at<double>(i,j) += sum_diff;

			// update epsi = max (sum_diff)
			if (abs(sum_diff) > epsi) epsi = abs(sum_diff);
		}

		//imwrite("output/HR_BP_" + int2str(iter) + ".bmp", HRimg);

		// check termination criterion
		// check termination criterion
		if (BPstop.type == TermCriteria::COUNT) {
			if (iter >= BPstop.maxCount) stop = true;
		}
		else if (BPstop.type == TermCriteria::EPS) {
			if (epsi < BPstop.epsilon) stop = true;	
		}
		else if (BPstop.type == TermCriteria::EPS + TermCriteria::COUNT) {
			if (iter >= BPstop.maxCount || epsi < BPstop.epsilon) stop = true;
		}

		iter++;
	}

	cout << "BP iteration: " << iter << endl;

	/**/

}

void BackProjection_Confidence ( Mat& HRimg, double scale, vector<Mat>& imgs, vector<Mat>& flows, Mat& PSF, Mat& BPk, TermCriteria& BPstop,  vector<Mat>& confidences)
{
	// all LR images should have same Size()

	// parameter
	double BP_c = 1;
	double interp_scale = 1000;

	int i, j, k, x, y;

	int LR_rows = imgs[0].rows;
	int LR_cols = imgs[0].cols;
	int HR_rows = LR_rows * scale;
	int HR_cols = LR_cols * scale;
	int LR_imgCount = imgs.size();

	// pre-interpolation
	Mat super_PSF;
	Mat super_BPk;	
	preInterpolation ( PSF, super_PSF, interp_scale);
	preInterpolation ( BPk, super_BPk, interp_scale);

	//imwrite("output/PSF.bmp", super_PSF*255);

	//----- for every pixel x of HR image, record influenced pixel y
	// initialize bucket
	HR_Pixel_Array* HR_pixels = new HR_Pixel_Array(HR_rows, HR_cols);
	// initialize influenced pixels (for each pixel in each LR img)
	LR_Pixel_Array* LR_pixels = new LR_Pixel_Array(LR_imgCount, LR_rows, LR_cols);
	//
	/*formInfluenceRelation (imgs,
							flows,
							LR_pixels,
							HR_pixels,
							scale,
							super_PSF,
							super_BPk,
							interp_scale);*/
	InfluenceRelation relations(imgs,
							flows,
							LR_pixels,
							HR_pixels,
							scale,
							super_PSF,
							super_BPk,
							interp_scale);
	
	// test bucket, forming HR initial guess
	HRimg = Mat::zeros(HR_rows, HR_cols, CV_64F);
	double tmp_sum;
	for (i = 0; i < HR_rows; i++) {
		for (j = 0; j < HR_cols; j++) {
			tmp_sum = 0;

			for (k = 0; k < HR_pixels->access(i, j).influence_link_cnt; k++) {
				tmp_sum += relations.influence_links[HR_pixels->access(i, j).influence_link_start + k].pixel->val;
			}
			if (HR_pixels->access(i, j).influence_link_cnt == 0) 
				HRimg.at<double>(i,j) = tmp_sum;
			else
				HRimg.at<double>(i,j) = tmp_sum / HR_pixels->access(i, j).influence_link_cnt;

			//for (k = 0; k < HR_pixels->access(i, j).influenced_pixels.size(); k++) {
			//	tmp_sum += HR_pixels->access(i, j).influenced_pixels[k].pixel->val;
			//}
			//if (HR_pixels->access(i, j).influenced_pixels.size() == 0) 
			//	HRimg.at<double>(i,j) = tmp_sum;
			//else
			//	HRimg.at<double>(i,j) = tmp_sum / HR_pixels->access(i, j).influenced_pixels.size();
		}
	}
	/**/
	
	double sum_diff, sum_hBP, sum_confidence;
	double cur_hBP, cur_confidence, diff;

	Mat LR_per;
	// start Back Projection
	
	int iter = 1;
	double epsi = 0;
	bool stop = false;
	while (!stop) {
		epsi = 0;

		HR_to_LR_percetion(HRimg, *LR_pixels, relations, super_PSF, true, interp_scale);

		// see LR perception
		//
		//LR_per = Mat::zeros(LR_rows, LR_cols, CV_64F);
		//for (k = 0; k < imgs.size(); k++) {
		//	for (i = 0; i < LR_rows; i++) for (j = 0; j < LR_cols; j++) {
		//		LR_per.at<double>(i,j) = LR_pixels[k][i][j].percetion;
		//	}
		//	String LR_p_name = "output/LR_P_" + int2str(k) + "_" + int2str(iter) + ".bmp";
		//	imwrite(LR_p_name, LR_per);
		//}
		//


		// for each HR x, calculate f(x)'n+1'
		for (i = 0; i < HR_rows; i++) for (j = 0; j < HR_cols; j++) {
			sum_diff = 0;
			sum_hBP = HR_pixels->access(i, j).hBP_sum;
			sum_confidence = 0;
			// for all influenced pixel
			for (k = 0; k < HR_pixels->access(i, j).influence_link_cnt; k++) {
				Influenced_Pixel& cur_influenced_pix = relations.influence_links[HR_pixels->access(i, j).influence_link_start + k];
				cur_hBP = cur_influenced_pix.hBP;
				cur_confidence = cur_influenced_pix.pixel -> confidence;
				// sum up hBP
				//sum_hBP += cur_hBP;
				// sum up confidence weight
				sum_confidence += cur_confidence;

				// sum up diff
				diff = cur_influenced_pix.pixel->val - cur_influenced_pix.pixel->perception;				
				sum_diff += diff * ( SQR(cur_hBP) ) * cur_confidence;	
				//sum_diff += diff *  cur_hBP  * cur_confidence;	
			}
			// deal with constant & weight normalization
			sum_diff /= (BP_c * sum_hBP * sum_confidence);

			// update HR
			HRimg.at<double>(i,j) += sum_diff;

			// update epsi = max (sum_diff)
			if (abs(sum_diff) > epsi) epsi = abs(sum_diff);
		}

		//imwrite("output/HR_BP_" + int2str(iter) + ".bmp", HRimg);

		// check termination criterion
		if (BPstop.type == TermCriteria::COUNT) {
			if (iter >= BPstop.maxCount) stop = true;
		}
		else if (BPstop.type == TermCriteria::EPS) {
			if (epsi < BPstop.epsilon) stop = true;	
		}
		else if (BPstop.type == TermCriteria::EPS + TermCriteria::COUNT) {
			if (iter >= BPstop.maxCount || epsi < BPstop.epsilon) stop = true;
		}

		iter++;
	}

	cout << "BP iteration: " << iter << endl;
	cout << "epsilon: " << epsi << endl;
	/**/

}

