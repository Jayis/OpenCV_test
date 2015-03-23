#include "DataStructures.h"

Pixel::Pixel()
{
	pos_x = 0;
	pos_y = 0;
	val = 0;
}

LR_Pixel::LR_Pixel()
{
	confidence = 0;
	perception = 0;
}

HR_Pixel::HR_Pixel()
{
	hBP_sum = 0;
}
//
HR_Pixel_Array::HR_Pixel_Array(int r, int c)
{
	hr_pixels = new HR_Pixel[r * c];
	HR_rows = r;
	HR_cols = c;
	HR_pixelCount = r * c;
}

HR_Pixel& HR_Pixel_Array::access(int i, int j)
{
	return hr_pixels[i * HR_cols + j];
}

HR_Pixel_Array::~HR_Pixel_Array()
{
	delete hr_pixels;
}

LR_Pixel_Array::LR_Pixel_Array(int k, int r, int c)
{
	lr_pixels = new LR_Pixel[k * r * c];
	LR_imgCount = k;
	LR_rows = r;
	LR_cols = c;
	LR_pixelCount = r * c;
}

LR_Pixel& LR_Pixel_Array::access(int k, int i, int j)
{
	return lr_pixels[k * LR_pixelCount + i * LR_cols + j];
}

LR_Pixel_Array::~LR_Pixel_Array()
{
	delete lr_pixels;
}
//
MySparseMat::MySparseMat()
{
	type = 0;
	nzcount = 0;
	rows = 0;
	cols = 0;
}

MySparseMat::MySparseMat(int r, int c, int t)
{
	type = t;
	nzcount = 0;
	rows = r;
	cols = c;

	if (type == 0) {
		elements.resize(r);
	}
	else {
		elements.resize(c);
	}
}

void MySparseMat::insertElement(Element& e)
{
	if (type == 0) {
		elements[e.i].push_back(e);
	}
	else {
		elements[e.j].push_back(e);
	}

	nzcount++;
}

void MySparseMat::setVal(int i, int j, double val)
{
	bool found = false;
	int k;

	if (type == 0) {
		for (k = 0; k < elements[i].size(); k++) {
			if (elements[i][k].j == j) {
				found = true;
				break;
			}
		}

		if (found) {
			elements[i][k].val = val;
		}
		else {
			insertElement(Element(i, j, val));
		}
	}
	else {
		for (k = 0; k < elements[j].size(); k++) {
			if (elements[j][k].i == i) {
				found = true;
				break;
			}
		}

		if (found) {
			elements[j][k].val = val;
		}
		else {
			insertElement(Element(i, j, val));
		}
	}
}

double MySparseMat::getVal(int i, int j)
{
	int k;
	bool found = false;

	if (type == 0) {
		for (k = 0; k < elements[i].size(); k++) {
			if (elements[i][k].j == j) {
				found = true;
				break;
			}
		}

		if (found) {
			return elements[i][k].val;
		}
		else {
			return 0;
		}
	}
	else {
		for (k = 0; k < elements[j].size(); k++) {
			if (elements[j][k].i == i) {
				found = true;
				break;
			}
		}

		if (found) {
			return elements[j][k].val;
		}
		else {
			return 0;
		}
	}
}

Element::Element(int ii, int jj, double value)
{
	i = ii;
	j = jj;
	val = value;
}