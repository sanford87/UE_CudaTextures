
#include "cuda_runtime.h"
#include "cuda_surface_types.h"
#include "cuda_runtime_api.h"
#include "surface_functions.h"
#include "device_launch_parameters.h"

#include <stdio.h>

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}


// CUDA surface kernel: write solid red into the texture
//__global__ void FillSurfaceKernel(cudaSurfaceObject_t surf, int width, int height)
//{
//    int x = blockIdx.x * blockDim.x + threadIdx.x;
//    int y = blockIdx.y * blockDim.y + threadIdx.y;
//
//    if (x < width && y < height)
//    {
//        uchar4 pixel = make_uchar4(255, 0, 255, 255); // RGBA8 red
//        surf2Dwrite(pixel, surf, x * sizeof(uchar4), y);
//    }
//}
__global__ void FillSurfaceKernel(cudaSurfaceObject_t surf, int width, int height, unsigned int seed)
{
	//int seed = 0; // You can modify this seed for different patterns
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        // Simple hash function for pseudo randomness
        unsigned int value = (x * 1973 + y * 9277 + seed * 26699) | 1;
        value ^= value << 13;
        value ^= value >> 17;
        value ^= value << 5;

        // Extract channels
        unsigned char r = (value & 0xFF);
        unsigned char g = (value >> 8) & 0xFF;
        unsigned char b = (value >> 16) & 0xFF;

        uchar4 pixel = make_uchar4(r, g, b, 255);
        surf2Dwrite(pixel, surf, x * sizeof(uchar4), y);
    }
}


// C wrapper function (called from Unreal)
cudaError_t LaunchFillSurfaceKernel(cudaArray_t array, int width, int height, unsigned int seed)
{
    cudaError_t cudaStatus = cudaSuccess;

    // Describe the surface
    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = array;

    // Create surface object
    cudaSurfaceObject_t surfObj = 0;
    cudaStatus = cudaCreateSurfaceObject(&surfObj, &resDesc);
    if (cudaStatus != cudaSuccess)
        return cudaStatus;

    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x,
        (height + blockDim.y - 1) / blockDim.y);

    // Launch kernel
    FillSurfaceKernel << <gridDim, blockDim >> > (surfObj, width, height, seed);

    // Check for kernel launch errors
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess)
    {
        cudaDestroySurfaceObject(surfObj);
        return cudaStatus;
    }

    // Wait for kernel to finish and check for runtime errors
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess)
    {
        cudaDestroySurfaceObject(surfObj);
        return cudaStatus;
    }

    // Destroy surface object
    cudaStatus = cudaDestroySurfaceObject(surfObj);
    return cudaStatus;
}