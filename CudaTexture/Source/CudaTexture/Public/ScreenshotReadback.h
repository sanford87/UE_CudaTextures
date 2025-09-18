// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "Engine/Texture2D.h"
#include "Widgets/SWindow.h"
#include "RenderGraphUtils.h"
#include "ImageCoreUtils.h"
#include "RHIGPUReadback.h"
#include "RHIResources.h"
#include "RHICommandList.h"
#include "RHI.h"
#include "ImageUtils.h"
#include "Delegates/Delegate.h"
#include "ScreenshotReadback.generated.h"

//DECLARE_LOG_CATEGORY_EXTERN(LogRenderingSamples, Log, All);
//DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnReadbackComplete, const TArray<uint8>&, PixelData);//Function definition for 'DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam' not found.
//DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnReadbackComplete);

typedef TRefCountPtr<FRHITexture2D> FTexture2DRHIRef;

UCLASS()
class CUDATEXTURE_API AScreenshotReadback : public AActor
{
	GENERATED_BODY()

public:
	// Sets default values for this actor's properties
	AScreenshotReadback();
	~AScreenshotReadback();

protected:
	// Called when the game starts or when spawned
	virtual void BeginPlay() override;

public:
	void TakeScreenshot();
	void TakeUIScreenshot();
	void ScreenshotViewportWithUI(SWindow& SlateWindow, const FTexture2DRHIRef& BackBuffer);
	void ScreenshotViewportWithoutUI(float InDeltaTime);
	inline void ScreenshotViewportWithoutUI() { ScreenshotViewportWithoutUI(0); }
	void ReadbackRenderTarget(FRHICommandListImmediate& RHICmdList, const FTexture2DRHIRef& RenderTarget);
	void OutputPNGTexture();

	template <typename BufferType>
	inline void SaveTexture(BufferType& OutTexture)
	{
		/**This code takes the texture buffer, regardless of type(FColor, FLinearColor etc)
		and passes it to the FImageUtils functionality which converts to a PNG file
		The default filepath (below) is the root project directory, with the current date time as the filename*/
		const FString FilePath = FPaths::ProjectDir() + FDateTime::Now().ToString();
		FImageView Image(OutTexture.GetData(), RTSize.X, RTSize.Y);
		bool bIsScreenshotSaved = FImageUtils::SaveImageAutoFormat(*FilePath, Image);
	}

	FDelegateHandle OnBackBufferReadyToPresent;
	FSlateRenderer* SlateRenderer;
	FViewport* CurrentViewport;
	TUniquePtr<FRHIGPUTextureReadback> RenderTargetReadback;
	EPixelFormat RTPixelFormat;
	FIntPoint RTSize;
	int32 RowPitch;
	int32 BufferHeight;
	uint32 BlockSizeX;
	uint32 BlockSizeY;
	uint32 BlockBytes;
	TArray<uint8> DataOut;
};
