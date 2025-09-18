// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "Kismet/BlueprintFunctionLibrary.h"
#include "CudaBPFuncLib.generated.h"

/**
 * 
 */
UCLASS()
class CUDATEXTURE_API UCudaBPFuncLib : public UBlueprintFunctionLibrary
{
	GENERATED_BODY()
public:
    UFUNCTION(BlueprintCallable, Category = "CUDA")
    static bool CustomCudaFunc();
    UFUNCTION(BlueprintCallable, Category = "CUDA") static FString GetDirectxVersion();

    UFUNCTION(BlueprintCallable, Category = "CUDA") static bool CallAddWithCUDA(const TArray<int32>& A, const TArray<int32>& B, TArray<int32>& OutResult);
};
