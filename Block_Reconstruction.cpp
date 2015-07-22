#include "Block_Reconstruction.h"

Block_Constructor::Block_Constructor(vector<Mat>& imgs,
									 vector<Mat>& flows,
									 vector<Mat>& confs,
									 double scale,
									 Mat& PSF)
{
	// parameters
	BigLR_rows = imgs[0].rows;
	BigLR_cols = imgs[0].cols;
	BigHR_rows = BigLR_rows * scale;
	BigHR_cols = BigLR_cols * scale;
	LR_imgCount = imgs.size();
	tmp_scale = scale;

	double interp_scale = 512;
	Mat super_PSF, super_BPk;
	preInterpolation ( PSF, super_PSF, interp_scale);
	super_BPk = super_PSF;

	//--------------------------------
	// not sure how to set over lap pix
	overlappingPix = PSF.rows + 1;
	
	
	longSide = (BigHR_rows > BigHR_cols) ? BigHR_rows : BigHR_cols;
	totalBlocksCount = pow(4, floor(log(longSide/200.f)/log(2.0f))); // origin: ceil

	blockPerAxis = sqrt(totalBlocksCount);
	blockWidth = double(BigHR_cols)/blockPerAxis;
	blockHeight = double(BigHR_rows)/blockPerAxis;	

	dataChunks.resize(int(BigHR_rows / blockHeight) + 1);
	for (int i = 0; i < dataChunks.size(); i++) {
		dataChunks[i].resize(int(BigHR_cols / blockWidth) + 1);
	}

	
	for (int i = 0; i < dataChunks.size(); i++) for (int j = 0; j < dataChunks[i].size(); j++)
	{
		int rectWidth, rectHeight;
		if ((j+1)*blockWidth >= BigHR_cols)
		{
			rectWidth = BigHR_cols - (int)(j*blockWidth);
		}
		else
		{
			rectWidth = (int)((j+1)*blockWidth) - (int)(j*blockWidth);
		}
		if ((i+1)*blockHeight >= BigHR_rows)
		{
			rectHeight = BigHR_rows - (int)(i*blockHeight);
		}
		else
		{
			rectHeight = (int)((i+1)*blockHeight) - (int)(i*blockHeight);
		}
		dataChunks[i][j].inBigHR = Rect((int)j*blockWidth, (int)i*blockHeight, rectWidth, rectHeight);
		dataChunks[i][j].inSmallHR = Rect(overlappingPix, overlappingPix, rectWidth, rectHeight);
		dataChunks[i][j].leftBorder = j*blockWidth - overlappingPix;
		dataChunks[i][j].rightBorder = j*blockWidth + rectWidth + overlappingPix;
		dataChunks[i][j].upBorder = i*blockHeight - overlappingPix;
		dataChunks[i][j].downBorder = i*blockHeight + rectHeight + overlappingPix;
		dataChunks[i][j].SmallHR_rows = dataChunks[i][j].downBorder - dataChunks[i][j].upBorder;
		dataChunks[i][j].SmallHR_cols = dataChunks[i][j].rightBorder - dataChunks[i][j].leftBorder;
		
	}
	//----- copy data -----
	tmp_imgs = imgs;
	tmp_flows = flows;
	tmp_confs = confs;

	//-------------------------------------------
	construct(super_PSF,
		super_BPk,
		interp_scale);

	//-------------------------------------------
	
}

void Block_Constructor::gather_LR_pix(DataChunk& dataChunk, int blockRowIdx, int blockColIdx)
{
	double pos_x, pos_y;
	int cur_blockRowIdx, cur_blockColIdx, cur_crossBlockColIdx, cur_crossBlockRowIdx;
	int i, j, k;
	

	for (k = 0; k < LR_imgCount; k++) for (i = 0; i < BigLR_rows; i++) for(j = 0; j < BigLR_cols; j++)
	{
		Vec2f& tmp_flow = tmp_flows[k].at<Vec2f>(i,j);
		
		pos_x = (j + tmp_flow[0] + 0.5) * tmp_scale;
		pos_y = (i + tmp_flow[1] + 0.5 ) * tmp_scale;

		LR_Pixel cur_LR_pix;

		cur_LR_pix.i = i;
		cur_LR_pix.j = j;
		cur_LR_pix.k = k;
		cur_LR_pix.val = (double)tmp_imgs[k].at<uchar>(i,j);
		cur_LR_pix.pos_x = pos_x;
		cur_LR_pix.pos_y = pos_y;
		cur_LR_pix.confidence = tmp_confs[k].at<double>(i, j);

		cur_blockColIdx = (int)pos_x / blockWidth;
		cur_blockRowIdx = (int)pos_y / blockHeight;
		if (cur_blockRowIdx == blockRowIdx && cur_blockColIdx == blockColIdx) {
			dataChunk.data_LR_pix.push_back(cur_LR_pix);
			continue;
		}
		// for left or up pix
		cur_crossBlockColIdx = (int)(pos_x+overlappingPix) / blockWidth;
		cur_crossBlockRowIdx = (int)(pos_y+overlappingPix) / blockHeight;
		if (cur_crossBlockColIdx != cur_blockColIdx) {
			if (cur_blockRowIdx == blockRowIdx && cur_crossBlockColIdx == blockColIdx) {
				dataChunk.data_LR_pix.push_back(cur_LR_pix);
				continue;
			}
		}
		if (cur_crossBlockRowIdx != cur_blockRowIdx) {
			if (cur_blockColIdx == blockColIdx && cur_crossBlockRowIdx == blockRowIdx) {
				dataChunk.data_LR_pix.push_back(cur_LR_pix);
				continue;
			}
		}
		if (cur_crossBlockColIdx == blockColIdx && cur_crossBlockRowIdx == blockRowIdx) {
			dataChunk.data_LR_pix.push_back(cur_LR_pix);
			continue;
		}
		// for right or down pix
		cur_crossBlockColIdx = (int)(pos_x-overlappingPix) / blockWidth;
		cur_crossBlockRowIdx = (int)(pos_y-overlappingPix) / blockHeight;
		if (cur_crossBlockColIdx != cur_blockColIdx) {
			if (cur_blockRowIdx == blockRowIdx && cur_crossBlockColIdx == blockColIdx) {
				dataChunk.data_LR_pix.push_back(cur_LR_pix);
				continue;
			}
		}
		if (cur_crossBlockRowIdx != cur_blockRowIdx) {
			if (cur_blockColIdx == blockColIdx && cur_crossBlockRowIdx == blockRowIdx) {
				dataChunk.data_LR_pix.push_back(cur_LR_pix);
				continue;
			}
		}
		if (cur_crossBlockColIdx == blockColIdx && cur_crossBlockRowIdx == blockRowIdx) {
			dataChunk.data_LR_pix.push_back(cur_LR_pix);
			continue;
		}

	}

}


void Block_Constructor::construct(Mat& super_PSF,
							   Mat& super_BPk,
							   double interp_scale)
{
	time_t t0, t1;

	time(&t0);

	for (int i = 0; i < dataChunks.size(); i++) for (int j = 0; j < dataChunks[i].size(); j++)
	{
		if (dataChunks[i][j].inBigHR.height <= 0 || dataChunks[i][j].inBigHR.width <= 0)
			continue;

		cout << "constructing i: " << i << ", j: " << j << endl;
		// we delete it at linear_constructor
		gather_LR_pix(dataChunks[i][j], i, j);
		dataChunks[i][j].tmp_HR_pixels = new HR_Pixel_Array(dataChunks[i][j].SmallHR_rows, dataChunks[i][j].SmallHR_cols);
		dataChunks[i][j].tmp_relations = new InfluenceRelation(dataChunks[i][j], super_PSF, super_BPk, interp_scale);		
		
		Linear_Constructor linearConstructor(dataChunks[i][j]);
		linearConstructor.addRegularization_grad2norm(0.05);
		//linearConstructor.solve_by_CG();
		linearConstructor.solve_by_GradientDescent();
		linearConstructor.output(dataChunks[i][j].smallHR);
		//*/
		dataChunks[i][j].data_LR_pix.clear();
		dataChunks[i][j].data_LR_pix.shrink_to_fit();
		delete dataChunks[i][j].tmp_relations;
		delete dataChunks[i][j].tmp_HR_pixels;

		//imwrite("output/datachunk" + int2str(i) + int2str(j) + ".bmp", dataChunks[i][j].smallHR);
		//*/
	}

	time(&t1);

	cout << "construction time: " << difftime(t1, t0) << endl;
}

void Block_Constructor::output(Mat& HRimg)
{
	HRimg = Mat::zeros(BigHR_rows, BigHR_cols, CV_64F );

	for (int i = 0; i < dataChunks.size(); i++) for (int j = 0; j < dataChunks[i].size(); j++)
	{
		if (dataChunks[i][j].inBigHR.height <= 0 || dataChunks[i][j].inBigHR.width <= 0)
			continue;
		//cout << "copying...  i: " << i << ", j: " << j << endl;
		dataChunks[i][j].smallHR(dataChunks[i][j].inSmallHR).copyTo(HRimg(dataChunks[i][j].inBigHR));
	}
}