// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Pawn.h"
#include "Kismet/BlueprintFunctionLibrary.h"
#include "Engine/TextureRenderTarget2D.h"// For UTextureRenderTarget2D and RTF_RGBA8
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


	UFUNCTION(BlueprintCallable, Category = "CUDA") bool TryCudaCall();
	UFUNCTION(BlueprintCallable, Category = "CUDA") static UTextureRenderTarget2D* CreateCUDACompatibleRenderTarget(UObject* Outer, int32 Width, int32 Height);

	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category = "CUDA") UTextureRenderTarget2D* MyRenderTarget;//Blueprint Assignable only allow of multicast or something.
	UPROPERTY(BlueprintReadWrite, EditAnywhere, Category = "GPU Interop")
	FString SharedHandle;
};
