#include <opencv2\core\core.hpp>
#include <opencv2\video\video.hpp>
#include <opencv2\calib3d\calib3d.hpp>
#include <opencv2\features2d\features2d.hpp>
#include <opencv2\nonfree\features2d.hpp>

#include "Linear_Reconstruction.h"
#include "FlexISP_Reconstruction.h"
#include "BackProjection.h"
#include "Methods.h"

using namespace std;
using namespace cv;

void flow2H_test ();

void LinearConstruct_test ();

void FlexISP_test ();

void OptFlow_BP_test ();

void OptFlow_ConfBP_test ();