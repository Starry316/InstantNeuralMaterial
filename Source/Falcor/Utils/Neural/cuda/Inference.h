#include <cuda_fp16.h>
#include <cuda_runtime.h>
// inference without synthesis
void launchInferInt8(
    int* weight,
    int* packedInput,
    float* quantizationScales,
    cudaTextureObject_t HP,
    cudaTextureObject_t DP,
    cudaTextureObject_t UP,
    float* output,
    unsigned int width,
    unsigned int height,
    int* validMask, float uvScale,
    bool isHalfAccumulation
);

// inference with synthesis
void launchInferSyn(
    int* weight,
    int* packedInput,
    float* quantizationScales,
    cudaTextureObject_t HP,
    cudaTextureObject_t DP,
    cudaTextureObject_t UP,
    cudaTextureObject_t TP,
    cudaTextureObject_t InvP,
    float* sampleList,
    float* output,
    unsigned int width,
    unsigned int height,
    int* validMask, float uvScale,
    bool isHalfAccumulation
);

