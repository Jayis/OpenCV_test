#include <iostream>

#include <opencv2\core\core.hpp>
#include <opencv2\video\video.hpp>
#include <opencv2\highgui\highgui.hpp>

#include "Tools.h"
#include "Methods.h"

using namespace std;
using namespace cv;

void tracking ();

void drawLine( Mat img, Point start, Point end );