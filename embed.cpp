#include "embed.h"



void tracking () {
	vector<Mat> imgs;
	vector<Mat> preProsImgs;
	vector<Mat> flows;

	int n = 12;

	imgs.resize(n);
	flows.resize(n);
	preProsImgs.resize(n);

	for (int i = 0; i < n; i++) {
		Mat tmp = imread("emb/" + int2str(i) + ".jpg", CV_LOAD_IMAGE_GRAYSCALE);
		resize(tmp, imgs[i], Size(0, 0), 1.0/6, 1.0/6, INTER_CUBIC);
	}
	ImgPreProcess(imgs, preProsImgs);

	imwrite("emb/mark0.jpg", imgs[0]);

	Ptr<DenseOpticalFlow> OptFlow = createOptFlow_DualTVL1();

	for(int i = 0; i < n-1; i++) {
		cout << "flow: " << i << endl;
		OptFlow->calc(preProsImgs[i], preProsImgs[i+1], flows[i]);
	}

	Point2f start = Point2f(521, 170);
	Point2f end = Point2f(587, 392);

	drawLine(imgs[0], start, end);

	imwrite("emb/mark0.jpg", imgs[0]);

	for(int i = 0; i < n-1; i++) {
		Vec2f& tmpFlow = flows[i].at<Vec2f>((start + end) * 0.5);

		start = start+Point2f(tmpFlow);
		end = end+Point2f(tmpFlow);

		drawLine(imgs[i+1], start, end);
		imwrite("emb/mark" + int2str(i+1) + ".jpg", imgs[i+1]);
	}
}

void drawLine( Mat img, Point start, Point end ) {
	int thickness = 2;
	int lineType = 8;
	line( img,
        start,
        end,
        Scalar( 0, 0, 0 ),
        thickness,
        lineType );

	rectangle( img,
        start,
        end,
		Scalar( 0, 255, 255 ),
		thickness,
		lineType );
}