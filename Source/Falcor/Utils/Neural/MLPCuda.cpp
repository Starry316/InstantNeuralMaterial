#include "MLPCuda.h"
#include "Core/Error.h"
#include "Core/Platform/OS.h"
#include "Core/API/Device.h"
#include "IOHelper.h"
#include "cuda/Inference.h"
#include <fstream>
namespace Falcor
{
int packInt8x4(int x, int y, int z, int w)
{
    return (x & 0x000000ff) | ((y << 8) & 0x0000ff00) | ((z << 16) & 0x00ff0000) | ((w << 24) & 0xff000000);
}

MLPCuda::MLPCuda() {}
void MLPCuda::loadFP32(ref<Device> pDevice, std::string networkPath)
{
    std::vector<float> cudaWeight = readBinaryFile(networkPath.c_str());

    mpFp32Buffer = pDevice->createBuffer(
        cudaWeight.size() * sizeof(float),
        ResourceBindFlags::Shared | ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
        MemoryType::DeviceLocal,
        cudaWeight.data()
    );

    logInfo("[MLPCuda] Weight buffer size: " + std::to_string(cudaWeight.size()));

    std::vector<__half> cudaWeightFP16(cudaWeight.size());
    for (size_t i = 0; i < cudaWeight.size(); i++)
    {
        cudaWeightFP16[i] = __float2half(cudaWeight[i]);
    }

    mpFp16Buffer = pDevice->createBuffer(
        cudaWeightFP16.size() * sizeof(__half), ResourceBindFlags::Shared, MemoryType::DeviceLocal, cudaWeightFP16.data()
    );
}
void MLPCuda::loadInt8(ref<Device> pDevice, std::string networkPath)
{
    std::vector<float> int8Weight = readBinaryFile(networkPath.c_str());

    std::vector<int> int8WeightInt(int8Weight.size() / 4);
    for (size_t i = 0; i < int8WeightInt.size(); i++)
    {
        int8WeightInt[i] =
            packInt8x4((int)int8Weight[i * 4], (int)int8Weight[i * 4 + 1], (int)int8Weight[i * 4 + 2], (int)int8Weight[i * 4 + 3]);
    }

    mpInt8Buffer =
        pDevice->createBuffer(int8WeightInt.size() * sizeof(int), ResourceBindFlags::Shared, MemoryType::DeviceLocal, int8WeightInt.data());
    logInfo("[MLPCuda] QINT8 buffer size: " + std::to_string(int8Weight.size()));
    logInfo("[MLPCuda] QINT8 buffer  {} {} {} {}", int8Weight[0], int8Weight[1], int8Weight[2], int8Weight[3]);
}

void MLPCuda::inferInt8Syn(
    int* packedInput,
    float* quantizationScales,
    float* output,
    int width,
    int height,
    int* valid,
    float scale,
    bool isHalfAccumulation
)
{
    launchInferSyn(
        (int*)mpInt8Buffer->getGpuAddress(),
        packedInput,
        quantizationScales,
        mHTexObj,
        mDTexObj,
        mUTexObj,
        mTTexObj,
        mInvTexObj,
        (float*)mpSampleBuffer->getGpuAddress(),
        output,
        width,
        height,
        valid,
        scale,
        isHalfAccumulation
    );
}

void MLPCuda::inferInt8(
    int* packedInput,
    float* quantizationScales,
    float* output,
    int width,
    int height,
    int* valid,
    float scale,
    bool isHalfAccumulation
)
{
    launchInferInt8(
        (int*)mpInt8Buffer->getGpuAddress(),
        packedInput,
        quantizationScales,
        mHTexObj,
        mDTexObj,
        mUTexObj,
        output,
        width,
        height,
        valid,
        scale,
        isHalfAccumulation
    );
}

} // namespace Falcor
