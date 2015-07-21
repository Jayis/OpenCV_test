#include <opencv2\core\core.hpp>
#include <opencv2\video\video.hpp>
#include <opencv2\calib3d\calib3d.hpp>
#include <opencv2\features2d\features2d.hpp>
#include <opencv2\nonfree\features2d.hpp>
#include <opencv2\legacy\legacy.hpp>

#include <time.h>

#include "Linear_Reconstruction.h"
//#include "FlexISP_Reconstruction.h" // i've remove it from project, for convenient, since this one is never done
#include "BackProjection.h"
#include "Methods.h"
#include "ExampleBased_Reconstruction.h"
#include "Mod_tv1flow.h"
#include "SymmConfOptFlowTV1.h"
#include "GA_Reconstruction.h"
#include "BP_Reconstruction.h"
#include "Block_Reconstruction.h"

using namespace std;
using namespace cv;

void multiOptFlow_exp(String test_set = "");

void symmetricOptFlow_test();

void exampleBased_test ();

void test();

void flow2H_test ();

void LinearConstruct_test ();

void FlexISP_test ();

void OptFlow_BP_test ();

void OptFlow_ConfBP_test ();

void GA_test();