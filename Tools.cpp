#include "Tools.h"

string int2str(int i)
{
	string s;
	stringstream ss (s);
	ss << i;

	return ss.str();
};

void writeImgDiff(Mat& a, Mat& b, string& name)
{
	Mat Diff = abs(a-b);
	imwrite(name, Diff);
}