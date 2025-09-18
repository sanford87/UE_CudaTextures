// Fill out your copyright notice in the Description page of Project Settings.


#include "MyTexturePawn.h"

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
	Super::SetupPlayerInputComponent(PlayerInputComponent);

}

