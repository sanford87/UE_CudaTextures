// Fill out your copyright notice in the Description page of Project Settings.
#pragma once


#include "MyTexturePawn.h"
//Unreal
#include "RHICommandList.h"
#include "RHIResources.h"
#include "RenderCore.h"
#include "RenderGraphUtils.h"
#include "Engine/TextureRenderTarget2D.h"// For UTextureRenderTarget2D and RTF_RGBA8
#include "Misc/ScopeLock.h"
#include "Async/Async.h"
#include "RHIGPUReadback.h"
#include "RHI.h"


//Cuda
#include "cuda.h"
#include "cuda_runtime.h"
//#include "cudaD3D11.h"// used with cuda_d3d11_interop causes overlapping declarations
#include"cuda_d3d11_interop.h"

//directX
#include <d3d11.h>
//#include <dxgi.h>
//#include <D3D11RenderTarget.h>
//#include <RHICore.h>


cudaArray_t array = nullptr;


// Sets default values
AMyTexturePawn::AMyTexturePawn()
{
 	// Set this pawn to call Tick() every frame.  You can turn this off to improve performance if you don't need it.
	PrimaryActorTick.bCanEverTick = true;

}

// Called when the game starts or when spawned
void AMyTexturePawn::BeginPlay()
{
	Super::BeginPlay();
	
}

// Called every frame
void AMyTexturePawn::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);
	// Example usage: you could map every frame if desired (but manage synchronization carefully)
	// if (bCUDARegistered)
	// {
	//    MapRenderTargetToCUDA();
	//    // run kernels here (synchronous or asynchronous)
	//    UnmapRenderTargetFromCUDA();
	// }
}

// Called to bind functionality to input
void AMyTexturePawn::SetupPlayerInputComponent(UInputComponent* PlayerInputComponent)
{
	//Super::SetupPlayerInputComponent(PlayerInputComponent);

}


UTextureRenderTarget2D* AMyTexturePawn::CreateCUDACompatibleRenderTarget(UObject* Outer, int32 Width, int32 Height)
{
    if (!Outer) Outer = GetTransientPackage();

    UTextureRenderTarget2D* NewRT = NewObject<UTextureRenderTarget2D>(Outer);
    if (!NewRT)
    {
        return nullptr;
    }

    // Choose format compatible with CUDA: RTF_RGBA8 (8-bit each channel) is simple to start with.
    NewRT->RenderTargetFormat = RTF_RGBA8;
    NewRT->bAutoGenerateMips = false;
    NewRT->Filter = TF_Bilinear;
    NewRT->AddressX = TA_Clamp;
    NewRT->AddressY = TA_Clamp;
    NewRT->ClearColor = FLinearColor::Red;
    NewRT->bGPUSharedFlag = true;
    NewRT->NeverStream = false;
    NewRT->InitAutoFormat(Width, Height);

    // Ensure resource exists immediately on the rendering thread
    NewRT->UpdateResourceImmediate(true);

    return NewRT;
}

bool AMyTexturePawn::FetchNativeD3D11Texture()
{
    if (MyRenderTarget)
    {
        FTextureRenderTargetResource* RTRes = MyRenderTarget->GameThread_GetRenderTargetResource();
        if (RTRes)
        {
            FRHITexture* TextureRHI = RTRes->GetRenderTargetTexture();
            //TextureRHI is now valid
        }
    }

    if (!MyRenderTarget)
    {
        UE_LOG(LogTemp, Warning, TEXT("FetchNativeD3D11Texture: No MyRenderTarget set."));
        return false;
    }

    // We need to call GameThread_GetRenderTargetResource() from game thread to get RT resource
    FTextureRenderTargetResource* RTRes = MyRenderTarget->GameThread_GetRenderTargetResource();
    if (!RTRes)
    {
        UE_LOG(LogTemp, Warning, TEXT("FetchNativeD3D11Texture: RenderTargetResource is null."));
        return false;
    }

    // Get the RHI texture reference (this is a reference to an FRHITexture)
    FRHITexture* TextureRHI = RTRes->GetRenderTargetTexture();
    if (!TextureRHI)
    {
        UE_LOG(LogTemp, Warning, TEXT("FetchNativeD3D11Texture: GetRenderTargetTexture() returned null."));
        return false;
    }

    // We must pull the native D3D11 resource on the render thread to be safe.
    // ENQUEUE_RENDER_COMMAND will run lambda on render thread and we capture the pointer to NativeD3D11Texture.
    struct FGetNativeResourceData
    {
        FRHITexture* TextureRHI;
        void** OutNativePtr;
        FEvent* DoneEvent;
    };

    FEvent* DoneEvent = FPlatformProcess::GetSynchEventFromPool(true);
    void* OutNative = nullptr;

    ENQUEUE_RENDER_COMMAND(GetNative)
        ([TextureRHI, &OutNative, DoneEvent](FRHICommandListImmediate& RHICmdList)
            {
                // FRHITexture::GetNativeResource() is an RHI-specific function — on D3D11 it returns ID3D11Texture2D*
                // Use GetNativeResource() which returns void*. Cast to ID3D11Texture2D* when needed.
                void* NativeResource = TextureRHI->GetNativeResource(); // NOTE: may be null if not D3D11 RHI
                OutNative = NativeResource;
                DoneEvent->Trigger();
            });

    // Wait for render thread to finish fetching the native pointer
    DoneEvent->Wait();
    FPlatformProcess::ReturnSynchEventToPool(DoneEvent);

    if (!OutNative)
    {
        UE_LOG(LogTemp, Error, TEXT("FetchNativeD3D11Texture: Native resource is null. Check that you are running D3D11 RHI."));
        return false;
    }

    // Save it
    NativeD3D11Texture = OutNative;

    UE_LOG(LogTemp, Log, TEXT("FetchNativeD3D11Texture: Native ptr fetched: %p"), NativeD3D11Texture);
    return true;
}

ID3D11Resource* GetD3D11ResourceFromRT(UTextureRenderTarget2D* RT)
{
    if (!RT) return nullptr;

    FTextureRenderTargetResource* RTResource = RT->GameThread_GetRenderTargetResource();
    if (!RTResource) return nullptr;
    FTexture2DRHIRef TextureRHI = RTResource->GetRenderTargetTexture();
    if (!TextureRHI.IsValid()) return nullptr;
    
    // Cast the generic RHI reference to D3D11
    //return static_cast<ID3D11Resource*>(TextureRHI->GetNativeResource());// WORKS. but replaced with code below

    ID3D11Resource* D3DResource = nullptr;
    // Run on render thread
    ENQUEUE_RENDER_COMMAND(FetchNativeResource)(
        [TextureRHI, &D3DResource](FRHICommandListImmediate& RHICmdList)
        {
            D3DResource = static_cast<ID3D11Resource*>(TextureRHI->GetNativeResource());
        });

    // Block until the above command is executed
    FlushRenderingCommands();

    return D3DResource;
}

bool AMyTexturePawn::RegisterRenderTargetWithCUDA()
{
    if (bCUDARegistered)
    {
        UE_LOG(LogTemp, Warning, TEXT("Already registered with CUDA."));
        return true;
    }

    if (!MyRenderTarget)
    {
        UE_LOG(LogTemp, Error, TEXT("No render target to register."));
        return false;
    }

    // 1) Ensure we have the native D3D11 texture pointer
    if (!FetchNativeD3D11Texture())
    {
        UE_LOG(LogTemp, Error, TEXT("RegisterRenderTargetWithCUDA: Failed to fetch native D3D11 texture."));
        return false;
    }

    if (!NativeD3D11Texture)
    {
        UE_LOG(LogTemp, Error, TEXT("RegisterRenderTargetWithCUDA: NativeD3D11Texture is null."));
        return false;
    }

    // 2) Register with CUDA (use cudaGraphicsRegisterFlagsSurfaceLoadStore if you plan to use surfaces)
    //ID3D11Resource* D3D11Res = reinterpret_cast<ID3D11Resource*>(NativeD3D11Texture);
    //ID3D11Resource* D3D11Res = reinterpret_cast<ID3D11Resource*>(MyRenderTarget->GetNativeResource());
    //ID3D11Resource* D3D11Res = reinterpret_cast<ID3D11Resource*>(NativeRes);
    //ID3D11Resource* D3D11Res = reinterpret_cast<ID3D11Resource*>(MyRenderTarget->GetRenderTargetResource());//might not return the right thing. Need to go one level deeper. Added function to get D3D11Resource
    
    ID3D11Resource* D3D11Res = GetD3D11ResourceFromRT(MyRenderTarget);
    if (!D3D11Res)
    {
        UE_LOG(LogTemp, Error, TEXT("D3D11Res is null."));
        return false;
    }
    //Create Cuda Error for testing
    cudaError_t CudaErr = cudaSuccess;

    // Option flags: if you want CUDA kernels to write/read image memory directly using surface/texture,
    // include cudaGraphicsRegisterFlagsSurfaceLoadStore. For general map -> cudaArray use 0.
    unsigned int RegisterFlags = cudaGraphicsRegisterFlagsSurfaceLoadStore; // or cudaGraphicsRegisterFlagsNone

    CudaErr = cudaGraphicsD3D11RegisterResource(&CudaGraphicsResource, D3D11Res, RegisterFlags);
    if (CudaErr != cudaSuccess)
    {
        UE_LOG(LogTemp, Error, TEXT("cudaGraphicsD3D11RegisterResource failed: %d: %s"), (int)CudaErr, ANSI_TO_TCHAR(cudaGetErrorString(CudaErr)));
        CudaGraphicsResource = nullptr;
        return false;
    }

    bCUDARegistered = true;
    UE_LOG(LogTemp, Log, TEXT("Render target registered with CUDA successfully."));
    return true;
}



bool AMyTexturePawn::UnregisterRenderTargetFromCUDA()
{
    if (!bCUDARegistered)
    {
        UE_LOG(LogTemp, Warning, TEXT("UnregisterRenderTargetFromCUDA: Not registered."));
        return true;
    }

    // Ensure resource is unmapped first
    if (MappedCudaArray)
    {
        UnmapRenderTargetFromCUDA();
    }

    if (CudaGraphicsResource)
    {
        cudaError_t err = cudaGraphicsUnregisterResource(CudaGraphicsResource);
        if (err != cudaSuccess)
        {
            UE_LOG(LogTemp, Warning, TEXT("cudaGraphicsUnregisterResource returned %d: %s"), (int)err, ANSI_TO_TCHAR(cudaGetErrorString(err)));
            // continue clearing pointers
        }
        CudaGraphicsResource = nullptr;
    }

    bCUDARegistered = false;
    NativeD3D11Texture = nullptr;
    UE_LOG(LogTemp, Log, TEXT("Unregistered render target from CUDA."));
    return true;
}

bool AMyTexturePawn::MapRenderTargetToCUDA()
{
    if (!bCUDARegistered || !CudaGraphicsResource)
    {
        UE_LOG(LogTemp, Warning, TEXT("MapRenderTargetToCUDA: Not registered with CUDA."));
        return false;
    }

    cudaError_t err = cudaSuccess;

    // Map resource(s)
    err = cudaGraphicsMapResources(1, &CudaGraphicsResource, 0);
    if (err != cudaSuccess)
    {
        UE_LOG(LogTemp, Error, TEXT("cudaGraphicsMapResources failed: %d: %s"), (int)err, ANSI_TO_TCHAR(cudaGetErrorString(err)));
        return false;
    }

    // Get mapped array (subresource 0). For texture2D, use array.
    array = nullptr;
    err = cudaGraphicsSubResourceGetMappedArray(&array, CudaGraphicsResource, 0, 0);
    if (err != cudaSuccess)
    {
        UE_LOG(LogTemp, Error, TEXT("cudaGraphicsSubResourceGetMappedArray failed: %d: %s"), (int)err, ANSI_TO_TCHAR(cudaGetErrorString(err)));
        // Unmap to clean up
        cudaGraphicsUnmapResources(1, &CudaGraphicsResource, 0);
        MappedCudaArray = array;
        return false;
    }

    MappedCudaArray = array;
    //cudaArray_t MappedCudaArray = nullptr;

    // Now you have a cudaArray_t pointing to the texture memory. From here:
    //  - Create a cudaResourceDesc -> cudaTextureObject or cudaSurfaceObject
    //  - Or use cudaMemcpy2DFromArray etc. (but that copies — to stay on GPU avoid copying and use surface/texture objects)
    //
    UE_LOG(LogTemp, Log, TEXT("MapRenderTargetToCUDA: Mapped cudaArray %p"), (void*)MappedCudaArray);
    return true;
}

bool AMyTexturePawn::UnmapRenderTargetFromCUDA()
{
    if (!bCUDARegistered || !CudaGraphicsResource)
    {
        UE_LOG(LogTemp, Warning, TEXT("UnmapRenderTargetFromCUDA: Not registered."));
        return false;
    }

    // If you created a cudaSurfaceObject or texture object, destroy it here before unmapping.

    // Clear mapped array pointer
    MappedCudaArray = nullptr;

    cudaError_t err = cudaGraphicsUnmapResources(1, &CudaGraphicsResource, 0);
    if (err != cudaSuccess)
    {
        UE_LOG(LogTemp, Error, TEXT("cudaGraphicsUnmapResources failed: %d: %s"), (int)err, ANSI_TO_TCHAR(cudaGetErrorString(err)));
        return false;
    }

    UE_LOG(LogTemp, Log, TEXT("UnmapRenderTargetFromCUDA: Unmapped successfully."));
    return true;
}

bool AMyTexturePawn::TryCudaCall()
{
    if (!bCUDARegistered || !MappedCudaArray)
    {
        UE_LOG(LogTemp, Warning, TEXT("TryCudaCall: Must register and map before calling CUDA operations."));
        return false;
    }

    // Example: create a CUDA surface object or texture object from MappedCudaArray and launch kernel.
    // For brevity I'll show a simple example of copying from the array into a device linear pointer (this will copy).
    // For true zero-copy GPU-only access you should create a surface or texture object and launch kernels that write into it directly.

    // Example: create resourceDesc -> surfaceObject (user must adapt to their kernel)
    // Pseudocode outline (left to implement as per your kernel needs):
    //
    // cudaResourceDesc resDesc = {};
    // resDesc.resType = cudaResourceTypeArray;
    // resDesc.res.array.array = MappedCudaArray;
    // cudaSurfaceObject_t surfObj = 0;
    // cudaCreateSurfaceObject(&surfObj, &resDesc);
    //
    // Launch kernel that writes to surfObj (bind as argument)
    //
    // cudaDestroySurfaceObject(surfObj);
    //
    // For demonstration we'll just log success:
    UE_LOG(LogTemp, Log, TEXT("TryCudaCall: You can now create a cudaSurfaceObject/textureObject from MappedCudaArray and run kernels."));

    return true;
}

void AMyTexturePawn::ClearNativeResources()
{
    // Ensure unregistered
    if (bCUDARegistered)
    {
        UnregisterRenderTargetFromCUDA();
    }
    NativeD3D11Texture = nullptr;
    MappedCudaArray = nullptr;
}