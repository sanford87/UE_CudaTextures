// Fill out your copyright notice in the Description page of Project Settings.
#pragma once

#include "MyTexturePawn.h"
#include "cuda.h"
#include "RHICommandList.h"
#include "RHIResources.h"
//#include "RenderCore.h"
#include "RenderGraphUtils.h"
#include "Engine/TextureRenderTarget2D.h"// For UTextureRenderTarget2D and RTF_RGBA8

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

}

// Called to bind functionality to input
void AMyTexturePawn::SetupPlayerInputComponent(UInputComponent* PlayerInputComponent)
{
	//Super::SetupPlayerInputComponent(PlayerInputComponent);

}

bool AMyTexturePawn::TryCudaCall()
{

	return true;
}

UTextureRenderTarget2D* AMyTexturePawn::CreateCUDACompatibleRenderTarget(UObject* Outer, int32 Width, int32 Height)
{
	// Create a new TextureRenderTarget2D object
	UTextureRenderTarget2D* MyRenderTarget = NewObject<UTextureRenderTarget2D>(Outer);
	if (MyRenderTarget)
	{
		// Set the properties of the render target
		MyRenderTarget->RenderTargetFormat = RTF_RGBA8; // Use a format compatible with CUDA
		MyRenderTarget->Filter = TF_Bilinear;
		MyRenderTarget->AddressX = TA_Clamp;
		MyRenderTarget->AddressY = TA_Clamp;
		MyRenderTarget->ClearColor = FLinearColor::Black;
		MyRenderTarget->InitAutoFormat(Width, Height);
		MyRenderTarget->UpdateResourceImmediate(true); // Ensure the resource is created immediately
	}
	return MyRenderTarget;
}