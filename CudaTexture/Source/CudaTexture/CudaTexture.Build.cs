
using System.IO;
using UnrealBuildTool;

public class CudaTexture : ModuleRules
{
    private string project_root_path
    {
        get { return Path.Combine(ModuleDirectory, "../.."); }
    }
    public CudaTexture(ReadOnlyTargetRules Target) : base(Target)
	{
		PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;

        PublicDependencyModuleNames.AddRange(new string[] { "Core", "CoreUObject", "Engine", "InputCore", "RHI", "RenderCore", "D3D11RHI", "Slate", "SlateCore" });

        PrivateDependencyModuleNames.AddRange(new string[] { });

        /// === CUDA interop ===
        string CudaRoot = @"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8";
        string CudaIncl = Path.Combine(CudaRoot, "include");
        string CudaLibDir = Path.Combine(CudaRoot, "lib", Target.Platform == UnrealTargetPlatform.Win64 ? "x64" : "Win32");

        PublicIncludePaths.Add(CudaIncl);
        PublicAdditionalLibraries.Add(Path.Combine(CudaLibDir, "cudart.lib"));
        PublicAdditionalLibraries.Add(Path.Combine(CudaLibDir, "cuda.lib"));


        //Custom CUDA Libs
        string custom_cuda_include = "CUDALib/include";
        string custom_cuda_lib = "CUDALib/lib";
        
        PublicIncludePaths.Add(Path.Combine(project_root_path, custom_cuda_include));
        PublicAdditionalLibraries.Add(Path.Combine(project_root_path, custom_cuda_lib, "UE_Cuda_Lib_Test.lib"));
        //write out where we think the file is
        System.Console.WriteLine(Path.Combine(project_root_path, custom_cuda_lib, "UE_Cuda_Lib_Test.lib"));

        // Uncomment if you are using Slate UI
        // PrivateDependencyModuleNames.AddRange(new string[] { "Slate", "SlateCore" });

        // Uncomment if you are using online features
        // PrivateDependencyModuleNames.Add("OnlineSubsystem");

        // To include OnlineSubsystemSteam, add it to the plugins section in your uproject file with the Enabled attribute set to true
    }
}
