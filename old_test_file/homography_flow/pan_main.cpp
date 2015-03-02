#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\features2d\features2d.hpp>
#include <opencv2\calib3d\calib3d.hpp>
#include <vector>
#include <iostream>

using namespace cv;
using namespace std;

const float EXS=1e-5;
#define SQR(x) ((x)*(x))

void estimate_H ( vector<Mat>& output, vector<Mat>& input, int H_array_rows, int H_array_cols, double lambda ) {
	for(int i=0; i<input.size(); i++){
		//cout << input[i] << endl;
	}

	int input_size = input.size();
	
	vector <vector <int> > connection;
	vector <Mat> tmp[2];
	
	output.resize(input_size);
	connection.resize(input_size);
	tmp[0].resize(input_size);
	tmp[1].resize(input_size);

	int H_r = input[0].rows;
	int H_c = input[0].cols;

	for (int i=0; i<output.size(); i++) {
		output[i] = Mat::zeros( H_r, H_c, CV_64F);
		tmp[0][i] = Mat::ones( H_r, H_c, CV_64F);
		tmp[1][i] = Mat::ones( H_r, H_c, CV_64F);
	}
	// construct (A'A)x = A'b  and A'A will be a square matrix
	
	int index = 0;
	for (int i=0; i < H_array_rows; i++) {
		for (int j=0; j < H_array_cols; j++) {
			index = i*H_array_cols + j;

			if (i > 0)
				connection[index].push_back(index-H_array_cols);
			if (i < H_array_rows - 1)
				connection[index].push_back(index+H_array_cols);
			if (j > 0)
				connection[index].push_back(index-1);
			if (j < H_array_cols - 1)
				connection[index].push_back(index+1);
		}
	}
	/**/
	// true = 1, false = 0
	bool turn = false;
	double diff = 1e8;
	int loop = 0;
	while (diff > 1e-6) {
		/*
		cout <<"loop: " << loop++ << endl;
		cout << "difference: " << diff << endl;
		/**/
		diff = 0;
		
		vector <Mat>& tmp_H_old = tmp[turn];
		turn = !turn;
		vector <Mat>& tmp_H_new = tmp[turn];

		// ith H
		for (int i=0; i<input.size(); i++) {
			// (1+coneection_num*lambda)*H_new = R + lambda*(sigma_connect  H_old)
			input[i].copyTo(tmp_H_new[i]);
			for (int k=0; k<connection[i].size(); k++) {
				tmp_H_new[i] += lambda*tmp_H_old[connection[i][k]];
			}
			tmp_H_new[i] /= (1 + connection[i].size()*lambda);

			for (int r=0; r < H_r; r++) {
				for (int c=0; c < H_c; c++) {
					diff += abs( tmp_H_new[i].at<double>(r,c) - tmp_H_old[i].at<double>(r,c) );
				}
			}
		}
		diff /= (input.size() * H_r * H_c);
		/*
		for (int i = 0; i < input.size(); i++) {
			cout << input[i] << endl;
			cout << tmp_H_new[i] << endl;
			cout << tmp_H_old[i] << endl;
		}
		/**/
	}
	cout << "convergence difference: " << diff << endl;

	for (int i = 0; i < input.size(); i++)
		output[i] = tmp[turn][i];
}

double computeErrorOfH(Mat& H, std::vector<Point2f>& pointsOrigin, std::vector<Point2f>& pointsRef) {
	double error = 0;
	std::vector<Point2f> pointsAlign(pointsOrigin.size());
	perspectiveTransform( pointsOrigin, pointsAlign, H);
	for(int i=0; i<pointsAlign.size(); i++) {
		error = error + sqrt( SQR(pointsAlign[i].x-pointsRef[i].x) + SQR(pointsAlign[i].y-pointsRef[i].y) );
		//cout << pointsAlign[i].x << " " << pointsRef[i].x << " " << pointsOrigin[i].x  << " " << error<< endl;
	}
	return error;
}

int main() {

	Mat img1 = imread("input/input001.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	Mat img2 = imread("input/input002.bmp", CV_LOAD_IMAGE_GRAYSCALE);

	// get the image data
	int height = img1.rows;
	int width = img1.cols;
	int longSide = (height > width) ? height : width;
	int layerTotalNum = ceil(log(longSide/400.0f)/log(2.0f));
	int layerToDetectFeatures = layerTotalNum - 1;
	double scaleFactor = pow(0.5, layerToDetectFeatures);

	Mat img1DetectFeaturesLayer = Mat::zeros(height*scaleFactor, width*scaleFactor, img1.type());
	Mat img2DetectFeaturesLayer = Mat::zeros(height*scaleFactor, width*scaleFactor, img1.type());
	resize(img1, img1DetectFeaturesLayer, img1DetectFeaturesLayer.size(), 0, 0);
	resize(img2, img2DetectFeaturesLayer, img2DetectFeaturesLayer.size(), 0, 0);

	Ptr<cv::FeatureDetector> detector = FeatureDetector::create("HARRIS"); 
	Ptr<cv::DescriptorExtractor> descriptor = DescriptorExtractor::create("BRIEF"); 

	// detect keypoints
	std::vector<KeyPoint> keypoints1, keypoints2;
	detector->detect(img1DetectFeaturesLayer, keypoints1);
	detector->detect(img2DetectFeaturesLayer, keypoints2);

	// extract features
	Mat desc1, desc2;
	descriptor->compute(img1DetectFeaturesLayer, keypoints1, desc1);
	descriptor->compute(img2DetectFeaturesLayer, keypoints2, desc2);

	if(desc1.type()!=CV_32F) {
		desc1.convertTo(desc1, CV_32F);
		desc2.convertTo(desc2, CV_32F);
	}

	FlannBasedMatcher matcher;
	std::vector< DMatch > matches;
	matcher.match( desc1, desc2, matches );

	double max_dist = 0; double min_dist = 100;

	//-- Quick calculation of max and min distances between keypoints
	for( int i = 0; i < desc1.rows; i++ )
	{ double dist = matches[i].distance;
	if( dist < min_dist && abs(dist) > 0.001 ) min_dist = dist;  //avoid min_dist is zero
	if( dist > max_dist ) max_dist = dist;
	}

	printf("-- Max dist : %f \n", max_dist );
	printf("-- Min dist : %f \n", min_dist );

	//-- Draw only "good" matches (i.e. whose distance is less than 3*min_dist )
	std::vector< DMatch > good_matches;

	for( int i = 0; i < desc1.rows; i++ )
	{ if( matches[i].distance < 10*min_dist )
	{ good_matches.push_back( matches[i]); }
	}

	//-- Localize the object
  std::vector<Point2f> img1GoodFP;
  std::vector<Point2f> img2GoodFP;

  for( int i = 0; i < good_matches.size(); i++ )
  {
    //-- Get the keypoints from the good matches
    img1GoodFP.push_back( keypoints1[ good_matches[i].queryIdx ].pt );
    img2GoodFP.push_back( keypoints2[ good_matches[i].trainIdx ].pt );
  }

  cout << "good FP count:" << img1GoodFP.size() << " " << img2GoodFP.size() << endl;

	Mat img_matches;
	drawMatches( img1DetectFeaturesLayer, keypoints1, img2DetectFeaturesLayer, keypoints2,
		good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
		vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
	imshow("match", img_matches);
	waitKey(0);

	//compute pyramid
	std::vector<std::vector<Mat>> homographyPyramid;
	homographyPyramid.resize(layerTotalNum+1);

	for(int layerNum = layerTotalNum; layerNum >= 0; layerNum--) {
		std::vector<Mat> homographyInLayer;
		int diff = layerTotalNum-layerNum;
		int countOfH = pow(4, diff);
		homographyInLayer.resize(countOfH);
		homographyPyramid[layerNum] = homographyInLayer;
	}

	for(int layerNum = layerTotalNum; layerNum >= 0; layerNum--) {
		scaleFactor = pow(0.5, layerNum);
		Mat img1ThisLayer = Mat::zeros(height*scaleFactor, width*scaleFactor, img1.type());
		Mat img2ThisLayer = Mat::zeros(height*scaleFactor, width*scaleFactor, img1.type());
		resize(img1, img1ThisLayer, img1ThisLayer.size(), 0, 0);
		resize(img2, img2ThisLayer, img2ThisLayer.size(), 0, 0);

		std::vector<Point2f> img1GoodFPThisLayer;
		std::vector<Point2f> img2GoodFPThisLayer;
		if(layerNum < layerToDetectFeatures) {
			int diff = layerToDetectFeatures - layerNum;
			int upscaleFactor = pow(2, diff);

			for(int count=0; count<img1GoodFP.size(); count++) {
				Point2f origin = img1GoodFP[count];
				img1GoodFPThisLayer.push_back(Point2f(origin.x*upscaleFactor, origin.y*upscaleFactor));
				origin = img2GoodFP[count];
				img2GoodFPThisLayer.push_back(Point2f(origin.x*upscaleFactor, origin.y*upscaleFactor));
			}

			std::vector<Mat>& homographyInLayer = homographyPyramid[layerNum];
			double blockPerAxis = sqrt(homographyInLayer.size());
			double blockWidth = double(img1ThisLayer.cols)/blockPerAxis;
			double blockHeight = double(img1ThisLayer.rows)/blockPerAxis;
			int blockCount = 0;
			for( double y = 0; y < img1ThisLayer.rows; y += blockHeight )
			{
				for( double x =  0 ; x < img1ThisLayer.cols ; x += blockWidth )
				{
					int imgx = floor(x+EXS);
					int imgy = floor(y+EXS);
					int imgwidth = floor(blockWidth+EXS);
					int imgheight = floor(blockHeight+EXS);
					Rect rect = Rect ( imgx, imgy , imgwidth, imgheight );
					Mat img1Block = Mat(img1ThisLayer, rect);
					Mat img2Block = Mat(img2ThisLayer, rect);

					std::vector<Point2f> img1GoodFPThisBlock;
					std::vector<Point2f> img2GoodFPThisBlock;
					for(int count=0; count<img1GoodFPThisLayer.size(); count++) {
						Point2f fp = img1GoodFPThisLayer[count];
						if( fp.x >= imgx && fp.y >= imgy && fp.x < imgx+imgwidth && fp.y < imgy+imgheight ) {
							img1GoodFPThisBlock.push_back(fp);
							img2GoodFPThisBlock.push_back(img2GoodFPThisLayer[count]);
						}
					}
					cout << "good fp block count:" << img1GoodFPThisBlock.size() << endl;
					if(img1GoodFPThisBlock.size()>=8) {
						Mat H = findHomography( img1GoodFPThisBlock, img2GoodFPThisBlock, CV_RANSAC, 2 );
						//cout << H << endl;
						std::vector<Mat>& homographyInLastLayer = homographyPyramid[layerNum+1];
						int lastx = (blockCount % int(blockPerAxis)) / 2;
						int lasty = (blockCount / int(blockPerAxis)) / 2;
						Mat lastH = homographyInLastLayer[lasty*int(blockPerAxis/2)+lastx].clone();
						lastH.at<double>(0, 2) = lastH.at<double>(0, 2)*2;  //translation need scale
						lastH.at<double>(1, 2) = lastH.at<double>(1, 2)*2;

						double errorLast = computeErrorOfH(lastH, img1GoodFPThisBlock, img2GoodFPThisBlock);
						double errorNow = computeErrorOfH(H, img1GoodFPThisBlock, img2GoodFPThisBlock);
						cout << (errorNow < errorLast) << " error n:" << errorNow << " " <<errorLast << endl;
						if( errorNow < errorLast )
							homographyInLayer[blockCount] = H;
						else
							homographyInLayer[blockCount] = lastH;
					} else {
						std::vector<Mat>& homographyInLastLayer = homographyPyramid[layerNum+1];
						int lastx = (blockCount % int(blockPerAxis)) / 2;
						int lasty = (blockCount / int(blockPerAxis)) / 2;
						Mat lastH = homographyInLastLayer[lasty*int(blockPerAxis/2)+lastx];
						
						//cout << (lasty*int(blockPerAxis/2)+lastx) << endl;

						Mat thisH = lastH.clone();
						thisH.at<double>(0, 2) = thisH.at<double>(0, 2)*2;  //translation need scale
						thisH.at<double>(1, 2) = thisH.at<double>(1, 2)*2;
						homographyInLayer[blockCount] = thisH;
					}

					//cout << homographyInLayer[blockCount];
					blockCount++;
					//imshow ( "smallImages", Mat ( img1ThisLayer, rect ));
					//waitKey(0);
				}
			}
		} else if(layerNum > layerToDetectFeatures) {
			for(int count=0; count<img1GoodFP.size(); count++) {
				Point2f origin = img1GoodFP[count];
				img1GoodFPThisLayer.push_back(Point2f(origin.x/2.0f, origin.y/2.0f));
				origin = img2GoodFP[count];
				img2GoodFPThisLayer.push_back(Point2f(origin.x/2.0f, origin.y/2.0f));
			}

			Mat H = findHomography( img1GoodFPThisLayer, img2GoodFPThisLayer, CV_RANSAC, 2 );
			cout << H << endl;
			homographyPyramid[layerNum][0] = H;

			std::vector<Point2f> img1FPTemp= img1GoodFP;
			std::vector<Point2f> img2FPTemp= img2GoodFP;
			img1GoodFP.clear();
			img2GoodFP.clear();
			std::vector<Point2f> pointsAlign(img1FPTemp.size());
			perspectiveTransform( img1FPTemp, pointsAlign, H);
			for(int i=0; i<pointsAlign.size(); i++) {
				double error = sqrt( SQR(pointsAlign[i].x-img2FPTemp[i].x) + SQR(pointsAlign[i].y-img2FPTemp[i].y) );
				if(error < 2) {
					img1GoodFP.push_back(img1FPTemp[i]);
					img2GoodFP.push_back(img2FPTemp[i]);
				}
			}
			cout << img1GoodFP.size() << " " << img2GoodFP.size() << endl;
			//cout << "error:" << computeErrorOfH(H, img1GoodFPThisLayer, img2GoodFPThisLayer);
			//Mat img1Transform = img1.clone();
			//warpPerspective( img1, img1Transform, H, img1Transform.size());
			//imwrite("img1T.bmp", img1Transform);
		} else {
			std::vector<Mat>& homographyInLayer = homographyPyramid[layerNum];

			for(int count=0; count<img1GoodFP.size(); count++) {
				Point2f origin = img1GoodFP[count];
				img1GoodFPThisLayer.push_back(Point2f(origin.x, origin.y));
				origin = img2GoodFP[count];
				img2GoodFPThisLayer.push_back(Point2f(origin.x, origin.y));
			}

			double blockPerAxis = sqrt(homographyInLayer.size());
			double blockWidth = double(img1ThisLayer.cols)/blockPerAxis;
			double blockHeight = double(img1ThisLayer.rows)/blockPerAxis;
			int blockCount = 0;
			for( double y = 0; y < img1ThisLayer.rows; y += blockHeight )
			{
				for( double x =  0 ; x < img1ThisLayer.cols ; x += blockWidth )
				{
					int imgx = floor(x+EXS);
					int imgy = floor(y+EXS);
					int imgwidth = floor(blockWidth+EXS);
					int imgheight = floor(blockHeight+EXS);
					Rect rect = Rect ( imgx, imgy , imgwidth, imgheight );
					Mat img1Block = Mat(img1ThisLayer, rect);
					Mat img2Block = Mat(img2ThisLayer, rect);

					std::vector<Point2f> img1GoodFPThisBlock;
					std::vector<Point2f> img2GoodFPThisBlock;
					for(int count=0; count<img1GoodFPThisLayer.size(); count++) {
						Point2f fp = img1GoodFPThisLayer[count];
						if( fp.x >= imgx && fp.y >= imgy && fp.x < imgx+imgwidth && fp.y < imgy+imgheight ) {
							img1GoodFPThisBlock.push_back(fp);
							img2GoodFPThisBlock.push_back(img2GoodFPThisLayer[count]);
						}
					}
					cout << "good fp block count:" << img1GoodFPThisBlock.size() << endl;
					if(img1GoodFPThisBlock.size()>=8) {
						Mat H = findHomography( img1GoodFPThisBlock, img2GoodFPThisBlock, CV_RANSAC, 2 );
						//cout << H << endl;
						std::vector<Mat>& homographyInLastLayer = homographyPyramid[layerNum+1];
						int lastx = (blockCount % int(blockPerAxis)) / 2;
						int lasty = (blockCount / int(blockPerAxis)) / 2;
						Mat lastH = homographyInLastLayer[lasty*int(blockPerAxis/2)+lastx].clone();
						lastH.at<double>(0, 2) = lastH.at<double>(0, 2)*2;  //translation need scale
						lastH.at<double>(1, 2) = lastH.at<double>(1, 2)*2;

						double errorLast = computeErrorOfH(lastH, img1GoodFPThisBlock, img2GoodFPThisBlock);
						double errorNow = computeErrorOfH(H, img1GoodFPThisBlock, img2GoodFPThisBlock);
						cout << (errorNow < errorLast) << " error n:" << errorNow << " " <<errorLast << endl;
						if( errorNow < errorLast )
							homographyInLayer[blockCount] = H;
						else
							homographyInLayer[blockCount] = lastH;
					} else {
						std::vector<Mat>& homographyInLastLayer = homographyPyramid[layerNum+1];
						int lastx = (blockCount % int(blockPerAxis)) / 2;
						int lasty = (blockCount / int(blockPerAxis)) / 2;
						Mat lastH = homographyInLastLayer[lasty*int(blockPerAxis/2)+lastx];
						Mat thisH = lastH.clone();
						thisH.at<double>(0, 2) = thisH.at<double>(0, 2)*2;  //translation need scale
						thisH.at<double>(1, 2) = thisH.at<double>(1, 2)*2;
						homographyInLayer[blockCount] = thisH;
					}

					cout << homographyInLayer[blockCount] << endl;
					blockCount++;
					//imshow ( "smallImages", Mat ( img1ThisLayer, rect ));
					//waitKey(0);
				}
			}
		}
	}

	std::vector<Mat> wholeImgH = homographyPyramid[0];
	std::vector<Mat> finalH;
	double blockPerAxis = sqrt(wholeImgH.size());
	cout << blockPerAxis << " " << wholeImgH.size();
	
	estimate_H(finalH, wholeImgH, int(blockPerAxis), int(blockPerAxis), 0.1);

	cout << wholeImgH[0];

	Mat img1T = img1.clone();
	Mat img1TE = img1.clone();
	double blockWidth = double(img1.cols)/blockPerAxis;
	double blockHeight = double(img1.rows)/blockPerAxis;
	int blockCount = 0;
	for( double y = 0; y < img1.rows; y += blockHeight )
	{
		for( double x =  0 ; x < img1.cols ; x += blockWidth )
		{
			int imgx = floor(x+EXS);
			int imgy = floor(y+EXS);
			int imgwidth = floor(blockWidth+EXS);
			int imgheight = floor(blockHeight+EXS);
			Rect rect = Rect ( imgx, imgy , imgwidth, imgheight );
			Mat img1Block = Mat(img1, rect);
			Mat img1BlockT = img1Block.clone();
			Mat img1BlockTE = img1Block.clone();
			warpPerspective( img1Block, img1BlockT, wholeImgH[blockCount], img1BlockT.size());
			warpPerspective( img1Block, img1BlockTE, finalH[blockCount], img1BlockTE.size());

			for( int row=imgy; row<imgy+imgheight; row++) {
				for(int col=imgx; col<imgx+imgwidth; col++) {
					img1T.at<uchar>(row, col) = img1BlockT.at<uchar>(row-imgy, col-imgx);
					img1TE.at<uchar>(row, col) = img1BlockTE.at<uchar>(row-imgy, col-imgx);
				}
			}

			blockCount++;
		}
	}

	imshow("a", img1T);
	imshow("b", img1TE);
	imwrite("output.bmp", img1T);
	imwrite("output2.bmp", img1TE);
	waitKey(0);
	return 0;
}