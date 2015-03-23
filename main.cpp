#include <opencv2\core\core.hpp>

#include "Experiments.h"

using namespace std;
using namespace cv;

int main() {
	//LinearConstruct_test ();

	writeImgDiff(imread("output/res2000_LinearConstruct_HR4_CG.bmp", CV_LOAD_IMAGE_GRAYSCALE),
		imread("Origin/resOri_01.bmp", CV_LOAD_IMAGE_GRAYSCALE),
		"output/res2000_OriginLinearConstruct4_Diff.bmp");

	system("pause");
}