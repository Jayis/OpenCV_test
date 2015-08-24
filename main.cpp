
#include <opencv2\core\core.hpp>

#include "Experiments.h"


using namespace std;
using namespace cv;

int tmp_blockPerAxis;
double construct_t[21], tmp_t;

int main() {
	
	LinearConstruct_test();

	for (int i = 1; i < 21; i++) {
		printf(" %2d * %2d construction time: %6.0f \n", i, i, construct_t[i]);
	}

	system("pause");
}
//*/