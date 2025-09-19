#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"


cudaError_t addWithCuda(int* c, const int* a, const int* b, unsigned int size);

// New: surface fill kernel launcher (defined in .cu file)
cudaError_t LaunchFillSurfaceKernel(cudaArray_t array, int width, int height);

