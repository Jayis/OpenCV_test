#include "BackProjection.h"

void BackProjection ( Mat& HRimg, double scale, vector<Mat>& imgs, vector<Mat>& flows, Mat& PSF, Mat& BPk, TermCriteria& BPstop )
{
	// all LR images should have same Size()

	// parameter
	double BP_c = 1;
	double interp_scale = 1000;

	int i , j, k, x, y;

	int LR_rows = imgs[0].rows;
	int LR_cols = imgs[0].cols;
	int HR_rows = LR_rows * scale;
	int HR_cols = LR_cols * scale;

	double pos_x, pos_y, bucket_center_x, bucket_center_y, dist_x, dist_y, dx, dy, offset_x, offset_y;
	int bucket_idx_i, bucket_idx_j, super_offset_x, super_offset_y;

	// pre-interpolation
	Mat super_PSF;
	Mat super_BPk;	
	preInterpolation ( PSF, super_PSF, interp_scale);
	preInterpolation ( BPk, super_BPk, interp_scale);

	int PSF_radius_x = super_PSF.cols / interp_scale / 2;
	int PSF_radius_y = super_PSF.rows / interp_scale / 2;
	int BPk_radius_x = super_BPk.cols / interp_scale / 2;
	int BPk_radius_y = super_BPk.rows / interp_scale / 2;

	imwrite("output/PSF.png", super_PSF*255);

	//----- for every pixel x of HR image, record influenced pixel y
	// initialize bucket
	vector < vector < Influence_Bucket> >  influence_bucket;
	influence_bucket.resize(HR_rows);
	for (i = 0; i < influence_bucket.size(); i++) {
		influence_bucket[i].resize(HR_cols);
	}
	// initialize influenced pixels (for each pixel in each LR img)
	vector < vector < vector <Pixel> > > LR_pixels;
	LR_pixels.resize(imgs.size());
	for (k = 0; k < imgs.size(); k++) {
		LR_pixels[k].resize(LR_rows);
		for (i = 0; i < LR_rows; i++) {
			LR_pixels[k][i].resize(LR_cols);
		}
	}

	// star record
	// for each image
	for (k = 0; k < imgs.size(); k++) {
		// for each pixel
		for (i = 0; i < LR_rows; i++) {
			for (j = 0; j < LR_cols; j++) {
				Vec2f& tmp_flow = flows[k].at<Vec2f>(i,j);
				pos_x = (j + tmp_flow[0] + 0.5) * scale;
				pos_y = (i + tmp_flow[1] + 0.5 ) * scale;

				LR_pixels[k][i][j].val = (double)imgs[k].at<uchar>(i,j);
				LR_pixels[k][i][j].pos_x = pos_x;
				LR_pixels[k][i][j].pos_y = pos_y;
				LR_pixels[k][i][j].i = i;
				LR_pixels[k][i][j].j = j;
				LR_pixels[k][i][j].k = k;

				// add to those buckets within radius
				// for each possible bucket
				for (y = -PSF_radius_y-1; y < PSF_radius_y + 3; y++) {
					for (x = -PSF_radius_x-1; x < PSF_radius_y + 3; x++) {
						bucket_idx_i = pos_y + y;
						bucket_idx_j = pos_x + x;
						bucket_center_x = bucket_idx_j + 0.5;
						bucket_center_y = bucket_idx_i + 0.5;
						// check if bucket exist
						if (bucket_center_x < 0 || bucket_center_y < 0 || bucket_center_y >= HR_rows || bucket_center_x >= HR_cols)
							continue;
						// check if within PSF_radius
						dx = pos_x - bucket_center_x;
						dy = pos_y - bucket_center_y;
						dist_x = abs(dx);
						dist_y = abs(dy);
						if (dist_x-0.5 > PSF_radius_x || dist_y-0.5 > PSF_radius_y)
							continue;
						// create a influence relation
						Influenced_Pixel tmp_pix;
						tmp_pix.pixel = &(LR_pixels[k][i][j]);
						//----- hbp
						offset_x = dx + BPk_radius_x + 0.5;
						offset_y = dy + BPk_radius_y + 0.5;
						// if offset is just on the edge of PSF
						if (offset_x == BPk_radius_x * 2 + 1) offset_x -= EXsmall;
						if (offset_y == BPk_radius_y * 2 + 1) offset_y -= EXsmall;
						super_offset_x = offset_x * interp_scale;
						super_offset_y = offset_y * interp_scale;
						tmp_pix.hBP = super_BPk.at<double>(super_offset_x, super_offset_y);
						// add to bucket
						influence_bucket[ bucket_idx_i ][ bucket_idx_j ].influenced_pixels.push_back( tmp_pix );
					}
				}
			}
		}
	}

	// test bucket, forming HR initial guess

	HRimg = Mat::zeros(HR_rows, HR_cols, CV_64F);
	double tmp_sum;
	for (i = 0; i < HR_rows; i++) {
		for (j = 0; j < HR_cols; j++) {
			tmp_sum = 0;			
			for (k = 0; k < influence_bucket[i][j].influenced_pixels.size(); k++) {
				tmp_sum += influence_bucket[i][j].influenced_pixels[k].pixel->val;
			}
			if (influence_bucket[i][j].influenced_pixels.size() == 0) 
				HRimg.at<double>(i,j) = tmp_sum;
			else
				HRimg.at<double>(i,j) = tmp_sum / influence_bucket[i][j].influenced_pixels.size();
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

		HR_to_LR_percetion(HRimg, LR_pixels, scale, super_PSF, true, interp_scale);

		// see LR perception
		/*
		LR_per = Mat::zeros(LR_rows, LR_cols, CV_64F);
		for (k = 0; k < imgs.size(); k++) {
			for (i = 0; i < LR_rows; i++) for (j = 0; j < LR_cols; j++) {
				LR_per.at<double>(i,j) = LR_pixels[k][i][j].percetion;
			}
			String LR_p_name = "output/LR_P_" + int2str(k) + "_" + int2str(iter) + ".png";
			imwrite(LR_p_name, LR_per);
		}
		/**/


		// for each HR x, calculate f(x)'n+1'
		for (i = 0; i < HR_rows; i++) for (j = 0; j < HR_cols; j++) {
			sum_diff = 0;
			sum_hBP = 0;
			// for all influenced pixel
			// sum up hBP first
			for (k = 0; k < influence_bucket[i][j].influenced_pixels.size(); k++) {
				sum_hBP += influence_bucket[i][j].influenced_pixels[k].hBP;
			}
			// sum up diff
			for (k = 0; k < influence_bucket[i][j].influenced_pixels.size(); k++) {
				diff = influence_bucket[i][j].influenced_pixels[k].pixel->val - influence_bucket[i][j].influenced_pixels[k].pixel->percetion;
				cur_hBP = influence_bucket[i][j].influenced_pixels[k].hBP;
				sum_diff += diff * ( SQR(cur_hBP) / BP_c / sum_hBP);
			}

			// update HR
			HRimg.at<double>(i,j) += sum_diff;

			// update epsi = max (sum_diff)
			if (abs(sum_diff) > epsi) epsi = abs(sum_diff);
		}

		//imwrite("output/HR_BP_" + int2str(iter) + ".png", HRimg);

		// check termination criterion
		if (BPstop.type == TermCriteria::COUNT) {
			if (iter < BPstop.maxCount) iter++;
			else stop = true;
		}
		else if (BPstop.type == TermCriteria::EPS) {
			if (epsi < BPstop.epsilon) stop = true;			
		}
		else if (BPstop.type == TermCriteria::EPS + TermCriteria::COUNT) {
			if (iter < BPstop.maxCount) iter++;
			else stop = true;
			if (epsi < BPstop.epsilon) stop = true;			
		}
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

	int i , j, k, x, y;

	int LR_rows = imgs[0].rows;
	int LR_cols = imgs[0].cols;
	int HR_rows = LR_rows * scale;
	int HR_cols = LR_cols * scale;

	double pos_x, pos_y, bucket_center_x, bucket_center_y, dist_x, dist_y, dx, dy, offset_x, offset_y;
	int bucket_idx_i, bucket_idx_j, super_offset_x, super_offset_y;

	// pre-interpolation
	Mat super_PSF;
	Mat super_BPk;	
	preInterpolation ( PSF, super_PSF, interp_scale);
	preInterpolation ( BPk, super_BPk, interp_scale);

	int PSF_radius_x = super_PSF.cols / interp_scale / 2;
	int PSF_radius_y = super_PSF.rows / interp_scale / 2;
	int BPk_radius_x = super_BPk.cols / interp_scale / 2;
	int BPk_radius_y = super_BPk.rows / interp_scale / 2;

	//imwrite("output/PSF.png", super_PSF*255);

	//----- for every pixel x of HR image, record influenced pixel y
	// initialize bucket
	vector < vector < Influence_Bucket> >  influence_bucket;
	influence_bucket.resize(HR_rows);
	for (i = 0; i < influence_bucket.size(); i++) {
		influence_bucket[i].resize(HR_cols);
	}
	// initialize influenced pixels (for each pixel in each LR img)
	vector < vector < vector <Pixel> > > LR_pixels;
	LR_pixels.resize(imgs.size());
	for (k = 0; k < imgs.size(); k++) {
		LR_pixels[k].resize(LR_rows);
		for (i = 0; i < LR_rows; i++) {
			LR_pixels[k][i].resize(LR_cols);
		}
	}

	// star record
	// for each image
	for (k = 0; k < imgs.size(); k++) {
		// for each pixel
		for (i = 0; i < LR_rows; i++) {
			for (j = 0; j < LR_cols; j++) {
				Vec2f& tmp_flow = flows[k].at<Vec2f>(i,j);
				pos_x = (j + tmp_flow[0] + 0.5) * scale;
				pos_y = (i + tmp_flow[1] + 0.5 ) * scale;

				LR_pixels[k][i][j].val = (double)imgs[k].at<uchar>(i,j);
				LR_pixels[k][i][j].confidence = confidences[k].at<double>(i,j);
				LR_pixels[k][i][j].pos_x = pos_x;
				LR_pixels[k][i][j].pos_y = pos_y;
				LR_pixels[k][i][j].i = i;
				LR_pixels[k][i][j].j = j;
				LR_pixels[k][i][j].k = k;

				// add to those buckets within radius
				// for each possible bucket
				for (y = -PSF_radius_y-1; y < PSF_radius_y + 3; y++) {
					for (x = -PSF_radius_x-1; x < PSF_radius_y + 3; x++) {
						bucket_idx_i = pos_y + y;
						bucket_idx_j = pos_x + x;
						bucket_center_x = bucket_idx_j + 0.5;
						bucket_center_y = bucket_idx_i + 0.5;
						// check if bucket exist
						if (bucket_center_x < 0 || bucket_center_y < 0 || bucket_center_y >= HR_rows || bucket_center_x >= HR_cols)
							continue;
						// check if within PSF_radius
						dx = pos_x - bucket_center_x;
						dy = pos_y - bucket_center_y;
						dist_x = abs(dx);
						dist_y = abs(dy);
						if (dist_x-0.5 > PSF_radius_x || dist_y-0.5 > PSF_radius_y)
							continue;
						// create a influence relation
						Influenced_Pixel tmp_pix;
						tmp_pix.pixel = &(LR_pixels[k][i][j]);
						//----- hbp
						offset_x = dx + BPk_radius_x + 0.5;
						offset_y = dy + BPk_radius_y + 0.5;
						// if offset is just on the edge of PSF
						if (offset_x == BPk_radius_x * 2 + 1) offset_x -= EXsmall;
						if (offset_y == BPk_radius_y * 2 + 1) offset_y -= EXsmall;
						super_offset_x = offset_x * interp_scale;
						super_offset_y = offset_y * interp_scale;
						tmp_pix.hBP = super_BPk.at<double>(super_offset_x, super_offset_y);
						// add to bucket
						influence_bucket[ bucket_idx_i ][ bucket_idx_j ].influenced_pixels.push_back( tmp_pix );
					}
				}
			}
		}
	}

	// test bucket, forming HR initial guess

	HRimg = Mat::zeros(HR_rows, HR_cols, CV_64F);
	Mat tmp_HR;
	resize(imgs[0], tmp_HR, Size(HR_rows, HR_cols), 0, 0, INTER_CUBIC);

	
	double tmp_sum, tmp_d_sum;
	for (i = 0; i < HR_rows; i++) {
		for (j = 0; j < HR_cols; j++) {
			//HRimg.at<double>(i,j) = (double)tmp_HR.at<uchar>(i,j);
			
			tmp_sum = 0;
			tmp_d_sum = 0;
			for (k = 0; k < influence_bucket[i][j].influenced_pixels.size(); k++) {
				tmp_sum += influence_bucket[i][j].influenced_pixels[k].pixel->val /** influence_bucket[i][j].influenced_pixels[k].hBP /** influence_bucket[i][j].influenced_pixels[k].pixel->confidence/**/;
				/*tmp_d_sum += influence_bucket[i][j].influenced_pixels[k].hBP /** influence_bucket[i][j].influenced_pixels[k].pixel->confidence/**/;
			}
			if (influence_bucket[i][j].influenced_pixels.size() == 0) {
				HRimg.at<double>(i,j) = tmp_sum;
			}
			else {
				tmp_d_sum = influence_bucket[i][j].influenced_pixels.size();
				HRimg.at<double>(i,j) = tmp_sum / tmp_d_sum;
			}
			/**/
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

		HR_to_LR_percetion(HRimg, LR_pixels, scale, super_PSF, true, interp_scale);

		// see LR perception
		//
		//LR_per = Mat::zeros(LR_rows, LR_cols, CV_64F);
		//for (k = 0; k < imgs.size(); k++) {
		//	for (i = 0; i < LR_rows; i++) for (j = 0; j < LR_cols; j++) {
		//		LR_per.at<double>(i,j) = LR_pixels[k][i][j].percetion;
		//	}
		//	String LR_p_name = "output/LR_P_" + int2str(k) + "_" + int2str(iter) + ".png";
		//	imwrite(LR_p_name, LR_per);
		//}
		//


		// for each HR x, calculate f(x)'n+1'
		for (i = 0; i < HR_rows; i++) for (j = 0; j < HR_cols; j++) {
			sum_diff = 0;
			sum_hBP = 0;
			sum_confidence = 0;
			// for all influenced pixel
			for (k = 0; k < influence_bucket[i][j].influenced_pixels.size(); k++) {
				cur_hBP = influence_bucket[i][j].influenced_pixels[k].hBP;
				cur_confidence = influence_bucket[i][j].influenced_pixels[k].pixel ->confidence;
				// sum up hBP
				sum_hBP += cur_hBP;
				// sum up confidence weight
				sum_confidence += cur_confidence;

				// sum up diff
				diff = influence_bucket[i][j].influenced_pixels[k].pixel->val - influence_bucket[i][j].influenced_pixels[k].pixel->percetion;				
				sum_diff += diff * ( SQR(cur_hBP) ) * cur_confidence/**/;	
				//sum_diff += diff *  cur_hBP  * cur_confidence/**/;	
			}
			// deal with constant & weight normalization
			sum_diff /= (BP_c * sum_hBP * sum_confidence/**/);


			// update HR
			HRimg.at<double>(i,j) += sum_diff;

			// update epsi = max (sum_diff)
			if (abs(sum_diff) > epsi) epsi = abs(sum_diff);
		}

		//imwrite("output/HR_BP_" + int2str(iter) + ".png", HRimg);

		// check termination criterion
		if (BPstop.type == TermCriteria::COUNT) {
			if (iter < BPstop.maxCount) iter++;
			else stop = true;
		}
		else if (BPstop.type == TermCriteria::EPS) {
			if (epsi < BPstop.epsilon) stop = true;			
		}
		else if (BPstop.type == TermCriteria::EPS + TermCriteria::COUNT) {
			if (iter < BPstop.maxCount) iter++;
			else stop = true;
			if (epsi < BPstop.epsilon) stop = true;			
		}
	}

	cout << "BP iteration: " << iter << endl;

	/**/

}

void HR_to_LR_percetion ( Mat& HRimg, vector < vector < vector <Pixel> > >& LR_pixels, double scale, Mat& PSF, bool is_super_PSF, double PSF_scale )
{

	int PSF_radius_x, PSF_radius_y;

	Mat super_PSF;

	// directly use imresize to have a approximate pre-interpolation of PSF
	if (is_super_PSF) {
		super_PSF = PSF;
	}
	else {
		PSF_scale = 1000;
		preInterpolation (PSF, super_PSF, PSF_scale);
	}

	PSF_radius_x = super_PSF.cols / PSF_scale / 2;
	PSF_radius_y = super_PSF.rows / PSF_scale / 2;

	double pos_x, pos_y;
	int cur_pix_i, cur_pix_j;
	int locate_pixel_i, locate_pixel_j;
	double cur_pix_pos_x, cur_pix_pos_y; 
	double PSF_left, PSF_top;
	double offset_x, offset_y, dist_x, dist_y;
	int super_offset_x, super_offset_y;

	for (int k = 0; k < LR_pixels.size(); k++) {
		for (int i = 0; i < LR_pixels[k].size(); i++) {
			for (int j = 0; j < LR_pixels[k][i].size(); j++) {
				LR_pixels[k][i][j].percetion = 0;

				pos_y = LR_pixels[k][i][j].pos_y;
				pos_x = LR_pixels[k][i][j].pos_x;

				locate_pixel_i = pos_y;
				locate_pixel_j = pos_x;

				// for all posiible neighbor pixel
				for (int y = -PSF_radius_y-1; y < PSF_radius_y+3; y++ ) {
					for (int x = -PSF_radius_x-1; x < PSF_radius_x+3; x++) {
						cur_pix_pos_x = locate_pixel_j + 0.5 + x;
						cur_pix_pos_y = locate_pixel_i + 0.5 + y;
						cur_pix_i = cur_pix_pos_y;
						cur_pix_j = cur_pix_pos_x;

						// if pixel is out of bound, it means it can't be sampled
						if ( cur_pix_pos_x < 0 || cur_pix_pos_x >= HRimg.cols || cur_pix_pos_y < 0 || cur_pix_pos_y >= HRimg.rows )
							continue;

						dist_x = abs(pos_x - cur_pix_pos_x);
						dist_y = abs(pos_y - cur_pix_pos_y);
						// if dist is bigger than PSF_radius
						if (dist_x-0.5 > PSF_radius_x || dist_y-0.5 > PSF_radius_y)
							continue;

						PSF_left = cur_pix_pos_x - PSF_radius_x - 0.5;
						PSF_top = cur_pix_pos_y - PSF_radius_y - 0.5;

						offset_x = pos_x - PSF_left ;
						offset_y = pos_y - PSF_top ;

						// if offset is just on the edge of PSF
						if (offset_x == PSF_radius_x * 2 + 1) offset_x -= EXsmall;
						if (offset_y == PSF_radius_y * 2 + 1) offset_y -= EXsmall;

						super_offset_x = offset_x * PSF_scale;
						super_offset_y = offset_y * PSF_scale;

						LR_pixels[k][i][j].percetion += super_PSF.at<double>( super_offset_y, super_offset_x ) * (HRimg.at<double>(cur_pix_i, cur_pix_j)  );

					}
				}

			}
		}
	}
}
/**/

