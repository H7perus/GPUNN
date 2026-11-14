#pragma once

#include "glm/glm.hpp"
#include "glm/gtc/packing.hpp"
#include "vulkan/vulkan.hpp"
#include "GpuInterface/Types/Device.h";
#include <random>

size_t convertCoopVecMatrix(KE::VK::Device &device, vk::CooperativeVectorMatrixLayoutNV srcLayout, vk::CooperativeVectorMatrixLayoutNV dstLayout, void* srcData, void* dstData, int columnCount, int rowCount)
{
    vk::ConvertCooperativeVectorMatrixInfoNV convertInfo;
    size_t returnSize = 10000000;
    convertInfo.numColumns = columnCount;
    convertInfo.numRows    = rowCount;

    convertInfo.srcComponentType    = vk::ComponentTypeKHR::eFloat16;
    convertInfo.srcLayout           = srcLayout;
    convertInfo.srcStride           = columnCount * 2;
    convertInfo.srcSize             = columnCount * rowCount * 2;
    convertInfo.srcData.hostAddress = srcData;

    convertInfo.dstComponentType    = vk::ComponentTypeKHR::eFloat16;
    convertInfo.dstLayout           = dstLayout;
    convertInfo.dstStride           = columnCount * 2;
    convertInfo.pDstSize            = &returnSize;
    convertInfo.dstData.hostAddress = dstData;

    device.GetVkDevice().convertCooperativeVectorMatrixNV(&convertInfo);
    return returnSize;
}

size_t queryMatrixConvertSize(KE::VK::Device &device, vk::CooperativeVectorMatrixLayoutNV srcLayout, vk::CooperativeVectorMatrixLayoutNV dstLayout, int columnCount, int rowCount)
{
    return convertCoopVecMatrix(device, srcLayout, dstLayout, nullptr, nullptr, columnCount, rowCount);
}

void initialiseWeights(int srcLayerSize, int dstLayerSize, u16* targetPointer, int seed = 1337)
{
    std::random_device rd; // Will be used to obtain a seed for the random number engine
    std::mt19937       gen(seed);

    std::normal_distribution<> ND(0, sqrt(2.f / srcLayerSize));
    std::uniform_real_distribution<> UD(-0.5, 0.5);
    
    for(int i = 0; i < srcLayerSize * dstLayerSize; i++)
    {
        float value  = UD(gen);
        targetPointer[i] = glm::packHalf1x16(value);
    }

}

