#include <cuda_runtime.h>

#include "Macros.h"

int findCudaDevice();

int gpuGetMaxGflopsDeviceId();

int _ConvertSMVer2Cores(int major, int minor);