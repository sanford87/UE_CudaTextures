// Fill out your copyright notice in the Description page of Project Settings.


#include "CudaBPFuncLib.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_d3d11_interop.h>
//#include "UE_Cuda_Lib_Test.h"
//#include "E:\GitLocal\UE_CudaTextures\CudaTexture\CUDALib\include\UE_Cuda_Lib_Test.h"
#include "E:\GitLocal\UE_CudaTextures\CudaTexture\CUDALib\include\UE_CudaRuntimeLib.h"

FString UCudaBPFuncLib::GetDirectxVersion()
{
    FString RHIName = GDynamicRHI->GetName();
    if (RHIName == TEXT("D3D11"))
    {
        // DirectX 11 is being used
    }
    else if (RHIName == TEXT("D3D12"))
    {
        // DirectX 12 is being used
    }
    else
    {
        // Other RHI (e.g., Vulkan)
    }
    return RHIName;
}

bool UCudaBPFuncLib::CustomCudaFunc()
{
    cudaDeviceProp  prop;
    int dev;

    memset(&prop, 0, sizeof(cudaDeviceProp));
    prop.major = 1;
    prop.minor = 0;
    cudaChooseDevice(&dev, &prop);
    //set DX device
    // cudaGraphicsGLRegisterBuffer
    // cudaGraphicsMapResources
    // cudaGraphicsResourceGetMappedPointer
    // Run Kernel
    // cudaGraphicsUnmapResources



    // 
    // 
    //cudaDbuffer
    //cudaD3D11GetDevice()
    //D3D11CreateDevice();
    //ID3D11Device::CreateBuffer;
    //cudaGraphicsD3D11RegisterResource
    //cudaD3D11SetDirect3DDevice(ID3D11Device *pD3D11Device, int Device=1) //ln 89
    //cudaD3D11GetDevice(-1);

    //cudaError_t cudaD3D11SetDirect3DDevice(, dev); //ln 89
    return true;
}

bool UCudaBPFuncLib::CallAddWithCUDA(const TArray<int32>& A, const TArray<int32>& B, TArray<int32>& OutResult)
{
    int32 Size = A.Num();

    if (Size == 0 || Size != B.Num())
    {
        UE_LOG(LogTemp, Error, TEXT("Input arrays can't be empty and must have the same size"));
        return false;
    }
    //Prepare output array
    OutResult.SetNumZeroed(Size);

    cudaError_t Result = addWithCuda(OutResult.GetData(), A.GetData(), B.GetData(), static_cast<unsigned int>(Size));
    if (Result != cudaSuccess)
    {
        UE_LOG(LogTemp, Error, TEXT("CUDA addition failed: %s"), *FString(cudaGetErrorString(Result)));
        return false;
    }
    return true;

}