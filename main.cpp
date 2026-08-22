#include "BasicTypeAliases.h"
#include <iostream>
#include <random>

#include "GpuInterface/VkHelpers.h"

#include "GpuInterface/Types/DeviceManager.h"


#include "GpuInterface/ShaderCompile/SlangCompileContext.h"

#include "GpuInterface/Types/PipelineCompute.h"
#include "GpuInterface/Types/PipelineGraphics.h"

#include "GpuInterface/Types/Sampler.h"
#include "Gpuinterface/Types/Texture.h"

#include "GpuNNHelpers.h"

#include "VkBootstrap.h"
//#include "mnist/mnist_reader_less.hpp"

#include "DrawSquares.h"

#include "glm/glm.hpp"
#include "glm/gtc/packing.hpp"
#include "vulkan/vulkan.hpp"

#include <chrono>
#include <thread>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_include.h"

#include "SDL3/SDL.h"
#include "SDL3/SDL_Vulkan.h"

VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

using namespace KE;


int main()
{

    if (SDL_Init(SDL_INIT_VIDEO) < 0)
    {
        printf("SDL initialization failed: %s\n", SDL_GetError());
        return 1;
    }

    SDL_Window* window = SDL_CreateWindow("GPUNN",  // Window title
                                          1200,             // Width
                                          600,              // Height
                                          SDL_WINDOW_VULKAN // Flags
    );

    auto         vkboot_inst = createInstance();
    vk::Instance vk_inst     = vkboot_inst.instance;

    InitDispatcherForDLL(vk_inst);

    VK::ContextManager::Init();

    VK::ContextManager::SetVkInstance(vkboot_inst);
    vk::SurfaceKHR surface;
    bool           test = SDL_Vulkan_CreateSurface(window, vk_inst, nullptr, (VkSurfaceKHR*)&surface);

    VK::ContextManager::GetInstance().AddDevice(createDevice(vkboot_inst, surface));

    VK::Device& device = VK::ContextManager::GetInstance().GetDevice(0);

    VK::ContextManager::AddSwapchain(0, surface, 1200, 600, 2);


    VK::SlangCompileContext compileContext;


    SlangCompiledUnit textureDrawShader = compileContext.CompileShaderPath("../../../../Shaders/TextureDraw.slang");
    SlangCompiledUnit neuralTextureDrawShader =
        compileContext.CompileShaderPath("../../../../Shaders/NeuralTextureDraw.slang");

    VK::PipelineGraphics textureDraw(0, textureDrawShader);
    VK::PipelineGraphics neuralTextureDraw(0, neuralTextureDrawShader);


    SlangCompiledUnit weightTrainingShader = compileContext.CompileShaderPath("../../../../Shaders/SlangNN.slang");

    SlangCompiledUnit weightApplyShader = compileContext.CompileShaderPath("../../../../Shaders/SlangNNApply.slang");

    Slang::ComPtr<slang::IBlob> blob = weightTrainingShader.getTargetCode();

    std::ofstream file("SlangNN.spv", std::ios::binary);
    file.write((char*)blob.get()->getBufferPointer(), blob.get()->getBufferSize());
    file.close();

    const int NeuronsPerHidden = 32;
    const int HiddenLayers     = 2; // must be 1 or more
    const int InputSize        = 4;
    const int OutputSize       = 3;

    VK::PipelineCompute pipeline(0, weightTrainingShader);

    VK::PipelineCompute applyPipeline(0, weightApplyShader);

    int numWeights = InputSize * NeuronsPerHidden + NeuronsPerHidden * NeuronsPerHidden * (HiddenLayers - 1) +
                     NeuronsPerHidden * OutputSize;


    // Initialise weights!


    size_t sizes[3];

    size_t summedTrainingOptimalOffset = 0;

    struct RunMLP
    {
        int          imageBase  = 0;
        int          isTraining = 0;
        int          sizes[3];
        int          totalSize;
        float        learningRate;
        unsigned int frameSeed;
        int          batchSize;
    };


    vk::ConvertCooperativeVectorMatrixInfoNV convertInfo;

    sizes[0] =
        queryMatrixConvertSize(device, vk::CooperativeVectorMatrixLayoutNV::eRowMajor,
                               vk::CooperativeVectorMatrixLayoutNV::eTrainingOptimal, InputSize, NeuronsPerHidden);

    sizes[1] = queryMatrixConvertSize(device, vk::CooperativeVectorMatrixLayoutNV::eRowMajor,
                                      vk::CooperativeVectorMatrixLayoutNV::eTrainingOptimal, NeuronsPerHidden,
                                      NeuronsPerHidden);

    sizes[2] =
        queryMatrixConvertSize(device, vk::CooperativeVectorMatrixLayoutNV::eRowMajor,
                               vk::CooperativeVectorMatrixLayoutNV::eTrainingOptimal, NeuronsPerHidden, OutputSize);

    int totalWeightBufferSize = sizes[0] + NeuronsPerHidden * 2 +
                                (sizes[1] + NeuronsPerHidden * 2) * (HiddenLayers - 1) + sizes[2] + OutputSize * 2;

    auto weightBuffer =
        VK::Buffer(0, totalWeightBufferSize, vk::BufferUsageFlagBits::eStorageBuffer,
                   vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, true);

    auto adjustmentBuffer =
        VK::Buffer(0, totalWeightBufferSize, vk::BufferUsageFlagBits::eStorageBuffer,
                   vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, true);


    vertexBuffer = KE::VK::Buffer(0, sizeof(vertices), vk::BufferUsageFlagBits::eVertexBuffer,
                                  vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);

    indexBuffer = KE::VK::Buffer(0, sizeof(vertexIndices), vk::BufferUsageFlagBits::eIndexBuffer,
                                 vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);


    f32* mappedVertBuffer = (float*)vertexBuffer.map();
    memcpy(mappedVertBuffer, vertices, sizeof(vertices));
    vertexBuffer.unmap();

    i32* mappedIndexBuffer = (i32*)indexBuffer.map();
    memcpy(mappedIndexBuffer, vertexIndices, sizeof(vertexIndices));
    indexBuffer.unmap();

    KE::VK::Sampler testSampler = VK::Sampler(0, true);

    int            w;
    int            h;
    int            comp;
    unsigned char* image =
        stbi_load("../../../../slangstars.png", &w, &h, &comp, STBI_rgb_alpha);

    auto testTexture = KE::VK::Texture(0, w, h, vk::Format::eR8G8B8A8Unorm, vk::ImageUsageFlagBits::eSampled,
                                       vk::MemoryPropertyFlagBits::eDeviceLocal, true);

    if (!image)
    {
        std::string error = stbi_failure_reason();
    }


    testTexture.UploadPixels(image);


    int latentTexSize = 32;

    auto latentTexture = KE::VK::Texture(0, latentTexSize, latentTexSize, vk::Format::eR16G16B16A16Sfloat,
                                         vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eStorage,
                                         vk::MemoryPropertyFlagBits::eDeviceLocal, true);


    auto latentTrainingBuffer =
        KE::VK::Buffer(0, latentTexSize * latentTexSize * 4 * 4, vk::BufferUsageFlagBits::eStorageBuffer,
                       vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, true);

    auto adamTrainingBuffer =
        KE::VK::Buffer(0, (totalWeightBufferSize * 2 + latentTexSize * latentTexSize * 4 * 4) * 2,
                       vk::BufferUsageFlagBits::eStorageBuffer,
                       vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, true);

    u16* latentData = new u16[latentTexSize * latentTexSize * 4];

    std::random_device rd; // Will be used to obtain a seed for the random number engine
    std::mt19937       gen(1337);
    std::uniform_real_distribution<> UD(0, 1);

    for (int i = 0; i < latentTexSize * latentTexSize * 4; i++)
        latentData[i] = glm::packHalf1x16(UD(gen));


    float* adamPtr = (float*)adamTrainingBuffer.map();

    for (int i = 0; i < (totalWeightBufferSize * 2 + latentTexSize * latentTexSize * 4 * 4) * 2 / 4; i++)
    {
        adamPtr[i] = 0.0f;
    }
    adamTrainingBuffer.unmap();
    // latentTrainingBuffer.unmap();
    latentTexture.UploadPixels(latentData);



    u16* weightsPointer = new u16[numWeights];
    int  valoffset      = 0;

    initialiseWeights(InputSize, NeuronsPerHidden, weightsPointer);

    valoffset += InputSize * NeuronsPerHidden;
    for (int i = 0; i < HiddenLayers - 1; i++)
    {
        initialiseWeights(NeuronsPerHidden, NeuronsPerHidden, weightsPointer + valoffset);
        valoffset += NeuronsPerHidden * NeuronsPerHidden;
    }

    initialiseWeights(NeuronsPerHidden, OutputSize, weightsPointer + valoffset);

    char* weightTargetPointer = (char*)weightBuffer.map();

    convertCoopVecMatrix(device, vk::CooperativeVectorMatrixLayoutNV::eRowMajor,
                         vk::CooperativeVectorMatrixLayoutNV::eTrainingOptimal, weightsPointer, weightTargetPointer,
                         InputSize, NeuronsPerHidden);

    int inputOffset  = InputSize * NeuronsPerHidden;
    int outputOffset = sizes[0] + NeuronsPerHidden * 2;
    for (int i = 0; i < HiddenLayers - 1; i++)
    {
        convertCoopVecMatrix(device, vk::CooperativeVectorMatrixLayoutNV::eRowMajor,
                             vk::CooperativeVectorMatrixLayoutNV::eTrainingOptimal, weightsPointer + inputOffset,
                             weightTargetPointer + outputOffset, NeuronsPerHidden, NeuronsPerHidden);
        inputOffset += NeuronsPerHidden * NeuronsPerHidden;
        outputOffset += sizes[1] + NeuronsPerHidden * 2;
    }

    convertCoopVecMatrix(device, vk::CooperativeVectorMatrixLayoutNV::eRowMajor,
                         vk::CooperativeVectorMatrixLayoutNV::eTrainingOptimal, weightsPointer + inputOffset,
                         weightTargetPointer + outputOffset, NeuronsPerHidden, OutputSize);

    weightBuffer.unmap();



    vk::CommandBuffer commandBuffer;

    RunMLP MLPRunInfoPre;
    MLPRunInfoPre.sizes[0] = sizes[0];
    MLPRunInfoPre.sizes[1] = sizes[1];
    MLPRunInfoPre.sizes[2] = sizes[2];

    MLPRunInfoPre.totalSize =
        sizes[0] + (sizes[1] + NeuronsPerHidden * 2) * (HiddenLayers - 1) + sizes[2] + OutputSize * 2;

    MLPRunInfoPre.imageBase  = 0;
    MLPRunInfoPre.isTraining = 1;
    vk::PushDataInfoEXT pushInfo{};
    pushInfo.offset       = 0;
    pushInfo.data.address = &MLPRunInfoPre;
    pushInfo.data.size    = sizeof(MLPRunInfoPre);


    convertCoopVecMatrix(device, vk::CooperativeVectorMatrixLayoutNV::eTrainingOptimal,
                         vk::CooperativeVectorMatrixLayoutNV::eRowMajor, (char*)adjustmentBuffer.map(), weightsPointer,
                         InputSize, NeuronsPerHidden);
    float val;
    for (int y = 0; y < NeuronsPerHidden; y++)
        for (int x = 0; x < InputSize; x++)
        {
            val = glm::unpackHalf1x16(weightsPointer[y * InputSize + x]) * 0.0002;
        }

    adjustmentBuffer.unmap();


    std::cout << std::dec;


    vk::ImageMemoryBarrier imgBarrier;
    imgBarrier.oldLayout        = vk::ImageLayout::eGeneral; // Or whatever you used
    imgBarrier.newLayout        = vk::ImageLayout::eGeneral; // Forces a metadata flush
    imgBarrier.srcAccessMask    = vk::AccessFlagBits::eShaderWrite;
    imgBarrier.dstAccessMask    = vk::AccessFlagBits::eShaderRead;
    imgBarrier.image            = latentTexture.GetVkImage(); // You need the VkImage handle here
    imgBarrier.subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};


    vk::MemoryBarrier barrier{vk::AccessFlagBits::eShaderWrite,
                              vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite};

    RunMLP MLPRunInfo;
    MLPRunInfo.sizes[0]  = sizes[0];
    MLPRunInfo.sizes[1]  = sizes[1];
    MLPRunInfo.sizes[2]  = sizes[2];
    MLPRunInfo.totalSize = sizes[0] + NeuronsPerHidden * 2 + (sizes[1] + NeuronsPerHidden * 2) * (HiddenLayers - 1) +
                           sizes[2] + OutputSize * 2;


    const int baseBatchSize = 4096;

    for (int epoch = 0; epoch < 10000000; epoch++)
    {


        std::cout << "PASS: " << epoch << std::endl;

        VK::ContextManager::GetSwapchain(0).BeginNextFrame();
        commandBuffer = VK::ContextManager::GetSwapchain(0).GetCurrentCommandBuffer();
        commandBuffer.bindResourceHeapEXT(device.GetResourceHeapBindInfoPtr());
        commandBuffer.bindSamplerHeapEXT(device.GetSamplerHeapBindInfoPtr());

        MLPRunInfo.imageBase    = epoch + 1;
        MLPRunInfo.isTraining   = 1; 
        MLPRunInfo.learningRate = 0.001; // * log(1 + fmin(epoch / 1000, 50));
        MLPRunInfo.frameSeed    = rand();
        MLPRunInfo.batchSize = baseBatchSize; // * pow(4, std::max(int(epoch / 5000), 4));
        vk::PushDataInfoEXT pushInfo{};
        pushInfo.offset       = 0;
        pushInfo.data.address = &MLPRunInfo;
        pushInfo.data.size    = sizeof(MLPRunInfo);


        commandBuffer.pushDataEXT(&pushInfo);


        commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, applyPipeline);
        commandBuffer.dispatch(ceil((MLPRunInfo.totalSize / 2 + latentTexSize * latentTexSize) / 64.f), 1, 1);

        commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands, vk::PipelineStageFlagBits::eAllCommands,
                                      {}, barrier, {}, {});

        commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline);
        commandBuffer.dispatch(MLPRunInfo.batchSize / 128, 1, 1);


        commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands, vk::PipelineStageFlagBits::eAllCommands,
                                      {}, {},    // No global barrier
                                      {},        // No buffer barrier
                                      imgBarrier // Use the image-specific barrier
        );
        Draw(commandBuffer, textureDraw.GetPipeline());
        Draw(commandBuffer, neuralTextureDraw.GetPipeline());


        commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands, vk::PipelineStageFlagBits::eAllCommands,
                                      {}, barrier, {}, {});

        VK::ContextManager::GetSwapchain(0).EndFrame();

        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }

    return 0;
}