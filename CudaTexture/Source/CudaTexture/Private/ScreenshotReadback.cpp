// Fill out your copyright notice in the Description page of Project Settings.

#pragma once
#include "ScreenshotReadback.h"
#include "Framework/Application/SlateApplication.h"
#include "Async/Async.h"
#include "RHISurfaceDataConversion.h"
#include "Kismet/KismetRenderingLibrary.h"

// Sets default values
AScreenshotReadback::AScreenshotReadback()
{
	// Set this actor to call Tick() every frame.  You can turn this off to improve performance if you don't need it.
	PrimaryActorTick.bCanEverTick = true;
	SlateRenderer = nullptr;
	//Ensure the FRHIGPUTextureReadback object is initialised
	RenderTargetReadback.Reset();
}

AScreenshotReadback::~AScreenshotReadback()
{
	//Clear the FRHIGPUTextureReadback
	RenderTargetReadback.Reset();
}

// Called when the game starts or when spawned
void AScreenshotReadback::BeginPlay()
{
	Super::BeginPlay();

	//Initialise the delegate object. 
	//This will allow us to hook in at the specific points of the frame where different types of screenshots can be taken.
	OnBackBufferReadyToPresent = FDelegateHandle();

	if (FSlateApplication::IsInitialized())
	{
		if (SlateRenderer == nullptr)
		{
			//Grab a reference to the slate renderer, just for one of the possible options for delegate call timings. Needed for the UI screenshot.
			SlateRenderer = FSlateApplication::Get().GetRenderer();
		}
	}

	// Initialize + register our component
	InputComponent = NewObject<UInputComponent>(this);
	InputComponent->RegisterComponent();
	if (InputComponent)
	{
		/** Bind inputs here - allow us to set keyboard keys for calling different types of screenshot calls
		 "InputComponent->BindAction("Test", IE_Pressed, this, &AActor::Test);"
		 Then use the project input settings to bind the actions to the specific keys etc. */
		InputComponent->BindAction("TakeScreenshot", IE_Pressed, this, &AScreenshotReadback::TakeScreenshot);
		InputComponent->BindAction("TakeUIScreenshot", IE_Pressed, this, &AScreenshotReadback::TakeUIScreenshot);

		// Bind this actor's input to the player controller
		EnableInput(GetWorld()->GetFirstPlayerController());
	}
}

void AScreenshotReadback::TakeScreenshot()
{
	if (GEngine)
	{
		CurrentViewport = GEngine->GameViewport->Viewport;
		if (CurrentViewport)
		{
			/**This delegate execute when the backbuffer texture has just finished preparing the scene,
			but prior to the UI elements being added. This allows us to make a copy of the scene backbuffer without any slate passes having executed yet*/
			if (FSlateApplication::IsInitialized())
			{
				OnBackBufferReadyToPresent = FSlateApplication::Get().OnPreTick().AddUObject(this, &AScreenshotReadback::ScreenshotViewportWithoutUI);
			}
			/** Below is a potential alternative call to use at for this location if the slate application is not being utilised at all in the project
			else
			{
				OnBackBufferReadyToPresent = FCoreDelegates::OnEndFrame.AddUObject(this, &AScreenshotReadback::ScreenshotViewportWithoutUI);
			}*/
		}
	}
}

void AScreenshotReadback::ScreenshotViewportWithUI(SWindow& SlateWindow, const FTexture2DRHIRef& BackBuffer)
{
	//Remove the delegate call as we only want it to execute once, not every frame
	if (OnBackBufferReadyToPresent.IsValid())
	{
		SlateRenderer->OnBackBufferReadyToPresent().Remove(OnBackBufferReadyToPresent);
		OnBackBufferReadyToPresent = FDelegateHandle();
	}

	/**The backbuffer object for this delegate, "OnBackBufferReadyToPresent" requires the backbuffer as an argument,
	so this can be passed directly into the readback function*/
	ENQUEUE_RENDER_COMMAND(BackBufferReadbackExecution)(
		[this, BackBuffer](FRHICommandListImmediate& RHICmdList)
		{
			ReadbackRenderTarget(RHICmdList, BackBuffer);
		});
}

void AScreenshotReadback::TakeUIScreenshot()
{
	if (SlateRenderer)
	{
		/** Unlike TakeScreenshot, this delegate hooks in at the end of the whole frame, after the slate / UI passes,
		allowing us to make a backbuffer copy which includes the scene + the slate elements*/
		OnBackBufferReadyToPresent = SlateRenderer->OnBackBufferReadyToPresent().AddUObject(this, &AScreenshotReadback::ScreenshotViewportWithUI);
	}
}

void AScreenshotReadback::ScreenshotViewportWithoutUI(float InDeltaTime)
{
	//Remove the delegate call as we only want it to execute once, not every frame
	if (OnBackBufferReadyToPresent.IsValid())
	{
		FSlateApplication::Get().OnPreTick().Remove(OnBackBufferReadyToPresent);
		OnBackBufferReadyToPresent = FDelegateHandle();
	}

	/**This version of the delegate, OnPreTick, does not supply the backbuffer from the function arguments,
	so the backbuffer reference must be retrieved manually, but the timing is correct for grabbing the scene without the UI*/
	ENQUEUE_RENDER_COMMAND(BackBufferReadbackExecution)(
		[this](FRHICommandListImmediate& RHICmdList)
		{
			FTexture2DRHIRef BackBuffer =
#if WITH_EDITOR
				CurrentViewport->GetRenderTargetTexture();
#else                
				RHICmdList.GetViewportBackBuffer(CurrentViewport->GetViewportRHI());
#endif

			ReadbackRenderTarget(RHICmdList, BackBuffer);
		});
}

void AScreenshotReadback::ReadbackRenderTarget(FRHICommandListImmediate& RHICmdList, const FTexture2DRHIRef& RenderTarget)
{
	/**First off, check there is a valid FRHIGPUTextureReadback object,
	or that it has a large enough staging texture to process the render target, and reset the object if not so it can be re-used.*/
	if (!RenderTargetReadback.IsValid() || RTSize != RenderTarget->GetSizeXY())
	{
		RenderTargetReadback.Reset(new FRHIGPUTextureReadback(TEXT("BackBufferReadback")));
	}

	/**Store the size and format for later use
	(the readback can sometimes return inaccurate values depending on platform due to rounding up of sizes to DWORDs,
	so we need to save the accurate pixel count for processing the screenshot*/
	RTSize = RenderTarget->GetSizeXY();
	RTPixelFormat = RenderTarget->GetFormat();
	/**This function takes the backbufferand creates a staging texture copy of it to readback,
	which means we are only briefly using the backbuffer on the GPU and other processes can continue to interact with it.
	This should handle all necessary transitions too except what the source texture moves on to, which may need to be declared explicitly afterwards*/
	RenderTargetReadback->EnqueueCopy(RHICmdList, RenderTarget);
	//Transition the backbuffer (source texture) back to readable (typically transitioning back to only Present is needed so this may be overkill)
	RHICmdList.Transition(FRHITransitionInfo(RenderTarget, ERHIAccess::CopySrc, ERHIAccess::ReadableMask));

	//Get the pixel format block sizes for processing the staging texture data
	FPixelFormatInfo PixelFormat = GPixelFormats[RTPixelFormat];
	BlockSizeX = PixelFormat.BlockSizeX;
	BlockSizeY = PixelFormat.BlockSizeY;
	BlockBytes = PixelFormat.BlockBytes;

	/** Separate RowPitch and BufferHeight values. Although they, in theory, do the same as the backbuffer size,
	due to the DWORD issue mentioned above, we need to be selective around how we use them,
	and default to the given backbuffer size where we can.*/
	RowPitch = 0;
	BufferHeight = 0;
	//Lock the readback staging texture and memcpy with the returned sizes. Includes flush logic so we can proceed,
	void* LockedData = RenderTargetReadback->Lock(RowPitch, &BufferHeight);
	int32 Count = (RowPitch * BlockSizeX * BlockBytes) * (BufferHeight * BlockSizeY);
	DataOut.AddUninitialized(Count);
	//Memcpy the staging texture data to the DataOut uint8 buffer and remove any dependency on the GPU from here
	FMemory::Memcpy(DataOut.GetData(), LockedData, Count);
	//Unlock the staging texture
	RenderTargetReadback->Unlock();

	/**To avoid clogging up the game thread, as we now have the data stored in a CPU buffer
	dispatch the writing of the screenshot file to a background thread instead, presuming timing isn't critical*/
	AsyncTask(ENamedThreads::AnyBackgroundThreadNormalTask, [this]
		{
			OutputPNGTexture();
		});
}

void AScreenshotReadback::OutputPNGTexture()
{
	//Create a new FColor buffer to process the output data into a format FImageUtils can save as a PNG file
	TArray<FColor> OutTexture;
	OutTexture.AddUninitialized(RTSize.X * RTSize.Y);

	/** This code processes the memcopy data currently in DataOut for use by FImageUtils to create a PNG output file.	*
	* There is definitely something to be said for us creating a GPU pass that prepares the data exactly as needed here instead, rather than relying on the CPU
	* Running this on a lower priority thread instead is currently best of a bad situation.
	*
	* The conversions handle the different formats of r.DefaultBackBufferPixelFormat.
	*
	* For simplicity, Alpha is overridden for each conversion to just be output at 100%, as several formats will return 0 in the AlphaChannel
	*
	* Note: Outputting to FLinearColor instead will mean FImageUtils creates and EXR instead of a PNG.
	* Alternative conversions can be found in RHISurfaceDataConversion.h, but note the alpha can not be automatically overriden with this code.
	*/
	switch (RTPixelFormat)
	{
	case PF_B8G8R8A8:
		for (int32 Y = 0; Y < RTSize.Y; Y++)
		{
			FColor* SrcPtr = (FColor*)(DataOut.GetData() + Y * RowPitch * BlockSizeX * BlockBytes);
			FColor* DestPtr = OutTexture.GetData() + Y * RTSize.X;
			for (int32 X = 0; X < RTSize.X; X++)
			{
				*DestPtr = FColor(
					SrcPtr->R,
					SrcPtr->G,
					SrcPtr->B,
					255
				);
				++SrcPtr;
				++DestPtr;
			}
		}
		break;
	case PF_FloatRGBA:
		for (int32 Y = 0; Y < RTSize.Y; Y++)
		{
			FFloat16Color* SrcPtr = (FFloat16Color*)(DataOut.GetData() + Y * RowPitch * BlockSizeX * BlockBytes);
			FColor* DestPtr = OutTexture.GetData() + Y * RTSize.X;
			for (int32 X = 0; X < RTSize.X; X++)
			{
				*DestPtr = SrcPtr->GetFloats().ToFColor(false);
				DestPtr->A = 255;
				++SrcPtr;
				++DestPtr;
			}
		}
		break;
	case PF_A2B10G10R10:
	default:
		for (int32 Y = 0; Y < RTSize.Y; Y++)
		{
			FRHIR10G10B10A2* SrcPtr = (FRHIR10G10B10A2*)(DataOut.GetData() + Y * RowPitch * BlockSizeX * BlockBytes);
			FColor* DestPtr = OutTexture.GetData() + Y * RTSize.X;
			for (int32 X = 0; X < RTSize.X; X++)
			{
				*DestPtr = (FLinearColor(
					(float)SrcPtr->R / 1023.0f,
					(float)SrcPtr->G / 1023.0f,
					(float)SrcPtr->B / 1023.0f,
					3.0f
				)).ToFColor(false);
				++SrcPtr;
				++DestPtr;
			}
		}
		break;
	}

	//Now that the data is processed, save the output texture as a PNG file
	SaveTexture<TArray<FColor>>(OutTexture);
}