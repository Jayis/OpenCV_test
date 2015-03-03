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