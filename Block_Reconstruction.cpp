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

	//HR_pixels = new HR_Pixel_Array(BigHR_rows, BigHR_cols);
	LR_pixels = new LR_Pixel_Array(LR_imgCount, BigLR_rows, BigLR_cols);
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

	// tmp variable
	double pos_x, pos_y;
	int blockRowIdx, blockColIdx, crossBlockColIdx, crossBlockRowIdx;

	dataChunks.resize(int(BigHR_rows / blockHeight) + 1);
	for (int i = 0; i < dataChunks.size(); i++) {
		dataChunks[i].resize(int(BigHR_cols / blockWidth) + 1);
	}

	for (int k = 0; k < LR_imgCount; k++) for (int i = 0; i < BigLR_rows; i++) for(int j = 0; j < BigLR_cols; j++)
	{
		Vec2f& tmp_flow = flows[k].at<Vec2f>(i,j);
		LR_Pixel* cur_LR_pix = &(LR_pixels->access(k, i, j));

		pos_x = (j + tmp_flow[0] + 0.5) * scale;
		pos_y = (i + tmp_flow[1] + 0.5 ) * scale;

		cur_LR_pix->val = (double)imgs[k].at<uchar>(i,j);
		cur_LR_pix->pos_x = pos_x;
		cur_LR_pix->pos_y = pos_y;
		cur_LR_pix->confidence = confs[k].at<double>(i, j);

		blockColIdx = (int)pos_x / blockWidth;
		blockRowIdx = (int)pos_y / blockHeight;
		dataChunks[blockRowIdx][blockColIdx].data_LR_pix.push_back(cur_LR_pix);
		// for right and down block
		crossBlockColIdx = (int)(pos_x+overlappingPix) / blockWidth;
		crossBlockRowIdx = (int)(pos_y+overlappingPix) / blockHeight;
		if (crossBlockColIdx != blockColIdx && crossBlockColIdx < dataChunks[0].size()) {
			dataChunks[blockRowIdx][crossBlockColIdx].data_LR_pix.push_back(cur_LR_pix);
		}
		if (crossBlockRowIdx != blockRowIdx && crossBlockRowIdx < dataChunks.size()) {
			dataChunks[crossBlockRowIdx][blockColIdx].data_LR_pix.push_back(cur_LR_pix);
		}
		if (crossBlockColIdx != blockColIdx && crossBlockRowIdx != blockRowIdx && crossBlockColIdx < dataChunks[0].size() && crossBlockRowIdx < dataChunks.size()) {
			dataChunks[crossBlockRowIdx][crossBlockColIdx].data_LR_pix.push_back(cur_LR_pix);
		}
		// for left and up block
		crossBlockColIdx = (int)(pos_x-overlappingPix) / blockWidth;
		crossBlockRowIdx = (int)(pos_y-overlappingPix) / blockHeight;
		if (crossBlockColIdx != blockColIdx && crossBlockColIdx >= 0) {
			dataChunks[blockRowIdx][crossBlockColIdx].data_LR_pix.push_back(cur_LR_pix);
		}
		if (crossBlockRowIdx != blockRowIdx && crossBlockRowIdx >= 0) {
			dataChunks[crossBlockRowIdx][blockColIdx].data_LR_pix.push_back(cur_LR_pix);
		}
		if (crossBlockColIdx != blockColIdx && crossBlockRowIdx != blockRowIdx && crossBlockColIdx >= 0 && crossBlockRowIdx >= 0) {
			dataChunks[crossBlockRowIdx][crossBlockColIdx].data_LR_pix.push_back(cur_LR_pix);
		}
	}
	for (int i = 0; i < dataChunks.size(); i++) for (int j = 0; j < dataChunks[i].size(); j++)
	{
		double rectWidth, rectHeight;
		if ((j+1)*blockWidth >= BigHR_cols)
		{
			rectWidth = BigHR_cols - j*blockWidth;
		}
		else
		{
			rectWidth = blockWidth;
		}
		if ((i+1)*blockHeight >= BigHR_rows)
		{
			rectHeight = BigHR_rows - i*blockHeight;
		}
		else
		{
			rectHeight = blockHeight;
		}
		dataChunks[i][j].inBigHR = Rect(j*blockWidth, i*blockHeight, rectWidth, rectHeight);
		dataChunks[i][j].inSmallHR = Rect(overlappingPix, overlappingPix, rectWidth, rectHeight);
		dataChunks[i][j].leftBorder = j*blockWidth - overlappingPix;
		dataChunks[i][j].rightBorder = j*blockWidth + rectWidth + overlappingPix;
		dataChunks[i][j].upBorder = i*blockHeight - overlappingPix;
		dataChunks[i][j].downBorder = i*blockHeight + rectHeight + overlappingPix;
		dataChunks[i][j].SmallHR_rows = rectHeight + 2*overlappingPix;
		dataChunks[i][j].SmallHR_cols = rectWidth + 2*overlappingPix;
		
	}
	//-------------------------------------------
	construct(super_PSF,
		super_BPk,
		interp_scale);

	//-------------------------------------------
	
}


void Block_Constructor::construct(Mat& super_PSF,
							   Mat& super_BPk,
							   double interp_scale)
{
	for (int i = 0; i < dataChunks.size(); i++) for (int j = 0; j < dataChunks[i].size(); j++)
	{
		if (dataChunks[i][j].inBigHR.height <= 0 || dataChunks[i][j].inBigHR.width <= 0)
			continue;
		// we delete it at linear_constructor
		dataChunks[i][j].tmp_HR_pixels = new HR_Pixel_Array(dataChunks[i][j].SmallHR_rows, dataChunks[i][j].SmallHR_cols);
		dataChunks[i][j].tmp_relations = new InfluenceRelation(dataChunks[i][j], super_PSF, super_BPk, interp_scale);		

		Linear_Constructor linearConstructor(dataChunks[i][j]);
		linearConstructor.addRegularization_grad2norm(0.05);
		linearConstructor.solve_byCG();
		linearConstructor.output(dataChunks[i][j].smallHR);

		//delete dataChunks[i][j].tmp_relations;
		//delete dataChunks[i][j].tmp_HR_pixels;

		//imwrite("output/datachunk" + int2str(i) + int2str(j) + ".bmp", dataChunks[i][j].smallHR);
		//*/
	}
}

void Block_Constructor::output(Mat& HRimg)
{
	HRimg = Mat::zeros(BigHR_rows, BigHR_cols, CV_64F );

	for (int i = 0; i < dataChunks.size(); i++) for (int j = 0; j < dataChunks[i].size(); j++)
	{
		if (dataChunks[i][j].inBigHR.height <= 0 || dataChunks[i][j].inBigHR.width <= 0)
			continue;
		cout << "copying...  i: " << i << ", j: " << j << endl;
		dataChunks[i][j].smallHR(dataChunks[i][j].inSmallHR).copyTo(HRimg(dataChunks[i][j].inBigHR));
	}
}