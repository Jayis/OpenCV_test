#include "NN_Reconstruction.h"

NN_Constructor::NN_Constructor( vector<Mat>& LR_imgs, vector<Mat>& flows, vector<Mat> confs, double scale, Mat& PSF)
{
	constructor(LR_imgs, flows, confs, scale, PSF);

	K = LR_imgCount;

	needRelease = true;
}

NN_Constructor::NN_Constructor( DataChunk& dataChunk )
{
	HR_rows = dataChunk.SmallHR_rows;
	HR_cols = dataChunk.SmallHR_cols;
	LR_pixelCount = dataChunk.data_LR_pix.size();
	
	relations = dataChunk.tmp_relations;
	HR_pixels = dataChunk.tmp_HR_pixels;

	rim = dataChunk.overLappingPix + 1;
	//rim = 0;
	K = 4;

	needRelease = false;
}

void NN_Constructor::constructor( vector<Mat>& LR_imgs, vector<Mat>& flows, vector<Mat> confs, double scale, Mat& PSF)
{
	int i, j, k;

	interp_scale = 512;

	LR_rows = LR_imgs[0].rows;
	LR_cols = LR_imgs[0].cols;
	HR_rows = LR_rows * scale;
	HR_cols = LR_cols * scale;
	LR_imgCount = LR_imgs.size();
	LR_pixelCount = LR_rows * LR_cols;
	HR_pixelCount = HR_rows * HR_cols;

	rim = PSF.rows + 1;

	// pre-interpolation
	Mat super_PSF, super_BPk;
	preInterpolation ( PSF, super_PSF, interp_scale);
	super_BPk = super_PSF;

	//----- for every pixel x of HR image, record influenced pixel y
	// initialize bucket
	cout << "alloc HR_pix_array\n";
	HR_pixels = new HR_Pixel_Array(HR_rows, HR_cols);
	// initialize influenced pixels (for each pixel in each LR img)
	cout << "alloc LR_pix_array\n";
	LR_pixels = new LR_Pixel_Array(LR_imgCount, LR_rows, LR_cols);
	//
	relations = new InfluenceRelation (LR_imgs,
							flows,
							LR_pixels,
							HR_pixels,
							scale,
							super_PSF,
							super_BPk,
							interp_scale,
							confs);

}

void NN_Constructor::solve()
{
	vector<Influenced_Pixel*> tmp;

	for (int i = 0; i < HR_rows; i++) for (int j = 0; j < HR_cols; j++)
	{
		HR_Pixel& cur_hr_pix = HR_pixels->access(i, j);

		tmp.clear();
		tmp.reserve(cur_hr_pix.influence_link_cnt);

		for (int idx = 0; idx < cur_hr_pix.influence_link_cnt; idx++)
		{
			tmp.push_back( &(relations->influence_links[cur_hr_pix.influence_link_start + idx]) );
		}
		
		sort(tmp.begin(), tmp.end(), compare_hpsf);

		cur_hr_pix.val = 0;
		double tmp_hBP_sum = 0;

		for (int k = 0; k < K && k < tmp.size(); k++)
		{
			tmp_hBP_sum += tmp[k]->hBP;
			cur_hr_pix.val += tmp[k]->hBP * tmp[k]->pixel->val;
		}
		cur_hr_pix.val /= tmp_hBP_sum;
	}	
}

void NN_Constructor::solve_by_LinearRefine(DataChunk& dataChunk)
{
	Mat var, varColor;
	showVarOfInfluencedPix(*HR_pixels,*relations, var);
	imwrite("output/var.bmp", var*125);

	DataChunk tmp_dataChunk;
	tmp_dataChunk.tmp_HR_pixels = HR_pixels;
	tmp_dataChunk.SmallHR_rows = HR_rows;
	tmp_dataChunk.SmallHR_cols = HR_cols;
	tmp_dataChunk.tmp_relations = relations;
	tmp_dataChunk.fullReconstruct = false;
		
	int highVar_idx = 0;
	for (int i = 0; i < HR_rows; i++) for (int j = 0; j < HR_cols; j++)
	{
		HR_Pixel& cur_hr_pix = HR_pixels->access(i, j);

		if (cur_hr_pix.var > 0.5) {
			cur_hr_pix.highVar_idx = highVar_idx;
			highVar_idx++;
		}
		else {
			cur_hr_pix.highVar_idx = -1;
		}
	}
	for (int r = 0; r < rim; r++) {
		for (int i = 1; i < HR_rows-1; i++) for (int j = 1; j < HR_cols-1; j++)
		{
			HR_Pixel& cur_hr_pix = HR_pixels->access(i, j);

			HR_Pixel& up_hr_pix = HR_pixels->access(i-1, j);
			HR_Pixel& down_hr_pix = HR_pixels->access(i+1, j);
			HR_Pixel& left_hr_pix = HR_pixels->access(i, j-1);
			HR_Pixel& right_hr_pix = HR_pixels->access(i, j+1);

			if (up_hr_pix.highVar_idx >= 0 ||
				down_hr_pix.highVar_idx >= 0 ||
				right_hr_pix.highVar_idx >= 0 ||
				left_hr_pix.highVar_idx >= 0)
			{
				if (cur_hr_pix.highVar_idx < 0) {
					cur_hr_pix.highVar_idx = -2;
				}				
				//cur_hr_pix.highVar_idx = highVar_idx;
				//highVar_idx++;
			}
		}
		//*/
		for (int i = 1; i < HR_rows-1; i++) for (int j = 1; j < HR_cols-1; j++)
		{
			HR_Pixel& cur_hr_pix = HR_pixels->access(i, j);

			if (cur_hr_pix.highVar_idx == -2)
			{
				cur_hr_pix.highVar_idx = highVar_idx;
				highVar_idx++;
			}
		}
	}
	tmp_dataChunk.highVar_cnt = highVar_idx;

	bool highVar;
	for (int idx = 0; idx < dataChunk.data_LR_pix.size(); idx++)
	//for (int k = 0; k < LR_imgCount; k++) for (int i = 0; i < LR_rows; i++) for (int j = 0; j < LR_cols; j++)
	{
		LR_Pixel& cur_lr_pix = dataChunk.data_LR_pix[idx];
		//LR_Pixel& cur_lr_pix = LR_pixels->access(k, i, j);

		highVar = false;

		for (int idx = 0; idx < cur_lr_pix.perception_link_cnt; idx++)
		{
			if (relations->perception_links[cur_lr_pix.perception_link_start + idx].pixel->highVar_idx != -1)
			{
				highVar = true;
				break;
			}
		}

		if(highVar)
		{
			tmp_dataChunk.data_LR_pix.push_back(cur_lr_pix);
		}
	}


	Linear_Constructor linearConstructor(tmp_dataChunk);
	linearConstructor.addRegularization_grad2norm(0.05);
	//linearConstructor.solve_by_CG_GPU();
	linearConstructor.solve_by_CG();
	//linearConstructor.solve_by_L2GradientDescent();
	//linearConstructor.solve_by_L2GradientDescent_GPU();
	linearConstructor.output(tmp_dataChunk.smallHR);

	imwrite("output/refine.bmp", tmp_dataChunk.smallHR);

	//
	solve();
	
	for(int i = 1; i < HR_rows-1; i++) for (int j = 1; j < HR_cols-1; j++)
	{
		HR_Pixel& cur_hr_pix = HR_pixels->access(i, j);

		if (cur_hr_pix.var > 0.5) {
			cur_hr_pix.val = tmp_dataChunk.smallHR.at<double>(i, j);
		}
	}
	/**/
}

bool compare_hpsf (Influenced_Pixel* a, Influenced_Pixel* b)
{ 
	return (a->hBP > b->hBP); 
}

void NN_Constructor::output(Mat& HRimg) {
	cout << "output as Mat\n";
	HRimg = Mat::zeros(HR_rows, HR_cols, CV_64F);
	int curIdx = 0;
	for (int i = 0; i < HR_rows; i++) for (int j = 0; j < HR_cols; j++) {
		HRimg.at<double>(i, j) = HR_pixels->access(i, j).val;

		curIdx ++;
	}
}

NN_Constructor::~NN_Constructor()
{
	if (needRelease) {
		delete HR_pixels;
		delete LR_pixels;
		delete relations;
	}
}