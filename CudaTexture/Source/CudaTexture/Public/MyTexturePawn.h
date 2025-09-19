// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Pawn.h"
#include "Kismet/BlueprintFunctionLibrary.h"
#include "Engine/TextureRenderTarget2D.h"// For UTextureRenderTarget2D and RTF_RGBA8
//Cuda
//#include "cuda.h"
//#include "cuda_runtime.h"
//#include "cudaD3D11.h"
//#include"cuda_d3d11_interop.h"

#include "MyTexturePawn.generated.h"

UCLASS()
class CUDATEXTURE_API AMyTexturePawn : public APawn
{
	GENERATED_BODY()

public:
	// Sets default values for this pawn's properties
	AMyTexturePawn();

protected:
	// Called when the game starts or when spawned
	virtual void BeginPlay() override;

public:	
	// Called every frame
	virtual void Tick(float DeltaTime) override;

	// Called to bind functionality to input
	virtual void SetupPlayerInputComponent(class UInputComponent* PlayerInputComponent) override;

    // Example Blueprint-callable wrapper to attempt a simple CUDA operation (user to fill kernel)
	UFUNCTION(BlueprintCallable, Category = "CUDA") bool TryCudaCall();
    //Create CUDA Render Target
	UFUNCTION(BlueprintCallable, Category = "CUDA") static UTextureRenderTarget2D* CreateCUDACompatibleRenderTarget(UObject* Outer, int32 Width =2048, int32 Height=2048);
    // Properties exposed to Blueprints
	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category = "CUDA") UTextureRenderTarget2D* MyRenderTarget;//Blueprint Assignable only allow of multicast or something.
    // Optionally store a string (handle or debug)
	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category = "GPU Interop") FString SharedHandle;

    // Expose whether resource is registered
    UPROPERTY(BlueprintReadOnly, Category = "CUDA")
    bool bCUDARegistered = false;

    // Blueprint-callable helpers;

    UFUNCTION(BlueprintCallable, Category = "CUDA")
    bool RegisterRenderTargetWithCUDA(); // registers native D3D resource with CUDA (call after RT created)

    UFUNCTION(BlueprintCallable, Category = "CUDA")
    bool UnregisterRenderTargetFromCUDA();

    // Map/unmap for per-frame access. Returns true on success.
    UFUNCTION(BlueprintCallable, Category = "CUDA")
    bool MapRenderTargetToCUDA();   // maps and prepares cudaArray for CUDA use

    UFUNCTION(BlueprintCallable, Category = "CUDA")
    bool UnmapRenderTargetFromCUDA();


private:
    // Internal native pointers + CUDA resource handle (only valid when bCUDARegistered true)
    // Forward declare CUDA types here as pointers so header compiles even without includes
    struct cudaGraphicsResource* CudaGraphicsResource = nullptr;
    void* NativeD3D11Texture = nullptr; // holds ID3D11Texture2D* as void* (casted)
    // when mapped:
    //struct cudaArray_t* MappedCudaArray = nullptr;
    //cudaArray_t MappedCudaArray = nullptr;

    //struct cudaGraphicsResource* CudaGraphicsResource = nullptr;
    //void* NativeD3D11Texture = nullptr;
    void* MappedCudaArray = nullptr; // opaque pointer, real type only used in cpp


    // Helper that pulls native resource (run on render thread)
    bool FetchNativeD3D11Texture();

    // Helper to clear native pointers
    void ClearNativeResources();
};
