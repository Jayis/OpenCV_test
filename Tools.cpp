#include "Tools.h"

string int2str(int i)
{
	string s;
	stringstream ss (s);
	ss << i;

	return ss.str();
};