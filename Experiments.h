#include <opencv2\core\core.hpp>
#include <opencv2\video\video.hpp>
#include <opencv2\calib3d\calib3d.hpp>
#include <opencv2\features2d\features2d.hpp>
#include <opencv2\nonfree\features2d.hpp>
#include <opencv2\legacy\legacy.hpp>

#include <time.h>

#include "Linear_Reconstruction.h"
//#include "FlexISP_Reconstruction.h" // i've remove it from project, for convenient, since this one is never done
//#include "BackProjection.h" // replace it with BP_reconstruction.h
#include "Methods.h"
//#include "ExampleBased_Reconstruction.h" // fail
#include "Mod_tv1flow.h"
#include "SymmConfOptFlowTV1.h"
#include "GA_Reconstruction.h"
#include "BP_Reconstruction.h"
#include "Block_Reconstruction.h"
#include "NN_Reconstruction.h"

using namespace std;
using namespace cv;

extern int tmp_blockPerAxis;
extern double construct_t[21], tmp_t;

void LinearConstruct_test ();

void FlexISP_test ();

void GA_test();