#include "Inference.h"
#include "Utils.h"

#define IN_NUM 24
#define IN_PACKED_NUM 6
#define HIDDEN_NUM 32
#define HIDDEN_PACKED_NUM 8
#define HALF_ACC false

__global__ void inferInt8Syn(
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
    int* validMask,
    float uvScale,
    bool isHalfAccumulation
)
{
    __shared__ int W[768];
    unsigned int localIdx = threadIdx.y * blockDim.x + threadIdx.x;
    if (localIdx < 256)
    {
        W[3 * localIdx] = weight[3 * localIdx];
        W[3 * localIdx + 1] = weight[3 * localIdx + 1];
        W[3 * localIdx + 2] = weight[3 * localIdx + 2];
    }
    __syncthreads();

    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;
    if (validMask[y * width + x] == 0)
        return;

    int val1[HIDDEN_NUM];
    int val2[HIDDEN_PACKED_NUM];

    float h1, h2;
    float d1, d2;
    float u, v;
    float u1, v1, u2, v2;
    unpackUnorm2x16(packedInput[5 * (y * width + x) + 0], h1, h2);
    unpackUnorm2x16(packedInput[5 * (y * width + x) + 1], d1, d2);
    unpackUnorm2x16(packedInput[5 * (y * width + x) + 2], u, v);
    unpackUnorm2x16(packedInput[5 * (y * width + x) + 3], u1, v1);
    unpackUnorm2x16(packedInput[5 * (y * width + x) + 4], u2, v2);

    float4 val = tex2DLayered<float4>(HP, h1, h2, 0);
    val2[0] = quantizeInt8x4f_safe(val, quantizationScales[0]);

    val = tex2DLayered<float4>(HP, h1, h2, 1);
    val2[1] = quantizeInt8x4f_safe(val, quantizationScales[0]);

    // =======================================
    // Synthesis
    u *= uvScale;
    v *= uvScale;
    float norm;
    float b0, b1, bs;

    b0 = B0cos(u, v);
    b1 = B1cos(u, v);
    bs = BSingularity(u, v);
    norm = b0 + b1 + bs;

    b0 /= norm;
    b1 /= norm;
    bs /= norm;

    norm = sqrt(b0 * b0 + b1 * b1 + bs * bs);

    float4 g0 = tex2DLayered<float4>(TP, v1, u1, 0);
    float4 g1 = tex2DLayered<float4>(TP, v2, u2, 0);
    float Gx, Gy, Gz, Gw;

    Gx = (g0.x - 0.5f) * b0 + (g1.x - 0.5f) * b1;
    Gy = (g0.y - 0.5f) * b0 + (g1.y - 0.5f) * b1;
    Gz = (g0.z - 0.5f) * b0 + (g1.z - 0.5f) * b1;
    Gw = (g0.w - 0.5f) * b0 + (g1.w - 0.5f) * b1;

    Gx = clampG(Gx / norm + 0.5f);
    Gy = clampG(Gy / norm + 0.5f);
    Gz = clampG(Gz / norm + 0.5f);
    Gw = clampG(Gw / norm + 0.5f);

    val.x = tex2DLayered<float4>(InvP, Gx, 0.0f, 0).x;
    val.y = tex2DLayered<float4>(InvP, Gy, 0.0f, 0).y;
    val.z = tex2DLayered<float4>(InvP, Gz, 0.0f, 0).z;
    val.w = tex2DLayered<float4>(InvP, Gw, 0.0f, 0).w;
    val2[2] = quantizeInt8x4f_safe(val, quantizationScales[0]);

    g0 = tex2DLayered<float4>(TP, v1, u1, 1);
    g1 = tex2DLayered<float4>(TP, v2, u2, 1);

    Gx = (g0.x - 0.5f) * b0 + (g1.x - 0.5f) * b1;
    Gy = (g0.y - 0.5f) * b0 + (g1.y - 0.5f) * b1;
    Gz = (g0.z - 0.5f) * b0 + (g1.z - 0.5f) * b1;
    Gw = (g0.w - 0.5f) * b0 + (g1.w - 0.5f) * b1;
    Gx = clampG(Gx / norm + 0.5f);
    Gy = clampG(Gy / norm + 0.5f);
    Gz = clampG(Gz / norm + 0.5f);
    Gw = clampG(Gw / norm + 0.5f);

    val.x = tex2DLayered<float4>(InvP, Gx, 0.0f, 1).x;
    val.y = tex2DLayered<float4>(InvP, Gy, 0.0f, 1).y;
    val.z = tex2DLayered<float4>(InvP, Gz, 0.0f, 1).z;
    val.w = tex2DLayered<float4>(InvP, Gw, 0.0f, 1).w;
    val2[3] = quantizeInt8x4f_safe(val, quantizationScales[0]);
    // =======================================

    val = tex2DLayered<float4>(DP, d1, d2, 0);
    val2[4] = quantizeInt8x4f_safe(val, quantizationScales[0]);

    val = tex2DLayered<float4>(DP, d1, d2, 1);
    val2[5] = quantizeInt8x4f_safe(val, quantizationScales[0]);

    // layer 1
    for (int k = 0; k < HIDDEN_NUM; k++)
    {
        val1[k] = 0;
        for (int j = 0; j < IN_PACKED_NUM; j++)
        {
            val1[k] = __dp4a(val2[j], W[k * IN_PACKED_NUM + j], val1[k]);
        }
    }

    for (int k = 0; k < HIDDEN_PACKED_NUM; k++)
    {
        if (isHalfAccumulation)
            // #if HALF_ACC
            val2[k] = quantizeInt8x4h_safe(
                dequantizeInt8h_relu(val1[4 * k], quantizationScales[1]),
                dequantizeInt8h_relu(val1[4 * k + 1], quantizationScales[1]),
                dequantizeInt8h_relu(val1[4 * k + 2], quantizationScales[1]),
                dequantizeInt8h_relu(val1[4 * k + 3], quantizationScales[1]),
                quantizationScales[2]
            );
        else
            // #else
            val2[k] = quantizeInt8x4f_safe(
                dequantizeInt8f_relu(val1[4 * k], quantizationScales[1]),
                dequantizeInt8f_relu(val1[4 * k + 1], quantizationScales[1]),
                dequantizeInt8f_relu(val1[4 * k + 2], quantizationScales[1]),
                dequantizeInt8f_relu(val1[4 * k + 3], quantizationScales[1]),
                quantizationScales[2]
            );
        // #endif
    }

    // layer 2
    for (int k = 0; k < HIDDEN_NUM; k++)
    {
        val1[k] = 0;
        for (int j = 0; j < HIDDEN_PACKED_NUM; j++)
        {
            val1[k] = __dp4a(val2[j], W[192 + k * HIDDEN_PACKED_NUM + j], val1[k]);
        }
    }
    for (int k = 0; k < HIDDEN_PACKED_NUM; k++)
    {
        if (isHalfAccumulation)
            // #if HALF_ACC
            val2[k] = quantizeInt8x4h_safe(
                dequantizeInt8h_relu(val1[4 * k], quantizationScales[3]),
                dequantizeInt8h_relu(val1[4 * k + 1], quantizationScales[3]),
                dequantizeInt8h_relu(val1[4 * k + 2], quantizationScales[3]),
                dequantizeInt8h_relu(val1[4 * k + 3], quantizationScales[3]),
                quantizationScales[4]
            );

        else
            // #else
            val2[k] = quantizeInt8x4f_safe(
                dequantizeInt8f_relu(val1[4 * k], quantizationScales[3]),
                dequantizeInt8f_relu(val1[4 * k + 1], quantizationScales[3]),
                dequantizeInt8f_relu(val1[4 * k + 2], quantizationScales[3]),
                dequantizeInt8f_relu(val1[4 * k + 3], quantizationScales[3]),
                quantizationScales[4]
            );
        // #endif
    }

    // layer 3
    for (int k = 0; k < HIDDEN_NUM; k++)
    {
        val1[k] = 0;
        for (int j = 0; j < HIDDEN_PACKED_NUM; j++)
        {
            val1[k] = __dp4a(val2[j], W[448 + k * HIDDEN_PACKED_NUM + j], val1[k]);
        }
    }
    for (int k = 0; k < HIDDEN_PACKED_NUM; k++)
    {
        if (isHalfAccumulation)
            // #if HALF_ACC
            val2[k] = quantizeInt8x4h_safe(
                dequantizeInt8h_relu(val1[4 * k], quantizationScales[5]),
                dequantizeInt8h_relu(val1[4 * k + 1], quantizationScales[5]),
                dequantizeInt8h_relu(val1[4 * k + 2], quantizationScales[5]),
                dequantizeInt8h_relu(val1[4 * k + 3], quantizationScales[5]),
                quantizationScales[6]
            );

        else
            // #else
            val2[k] = quantizeInt8x4f_safe(
                dequantizeInt8f_relu(val1[4 * k], quantizationScales[5]),
                dequantizeInt8f_relu(val1[4 * k + 1], quantizationScales[5]),
                dequantizeInt8f_relu(val1[4 * k + 2], quantizationScales[5]),
                dequantizeInt8f_relu(val1[4 * k + 3], quantizationScales[5]),
                quantizationScales[6]
            );
        // #endif
    }

    // final layer
    for (int k = 0; k < 3; k++)
    {
        val1[k] = 0;
        for (int j = 0; j < HIDDEN_PACKED_NUM; j++)
        {
            val1[k] = __dp4a(val2[j], W[704 + k * HIDDEN_PACKED_NUM + j], val1[k]);
        }
    }
    __syncthreads();
    if (isHalfAccumulation)
    {
        output[4 * (y * width + x) + 0] = dequantizeInt8h_relu(val1[0], quantizationScales[7]);
        output[4 * (y * width + x) + 1] = dequantizeInt8h_relu(val1[1], quantizationScales[7]);
        output[4 * (y * width + x) + 2] = dequantizeInt8h_relu(val1[2], quantizationScales[7]);
    }
    // #if HALF_ACC

    else
    {
        output[4 * (y * width + x) + 0] = dequantizeInt8f_relu(val1[0], quantizationScales[7]);
        output[4 * (y * width + x) + 1] = dequantizeInt8f_relu(val1[1], quantizationScales[7]);
        output[4 * (y * width + x) + 2] = dequantizeInt8f_relu(val1[2], quantizationScales[7]);
    }
    // #else

    // #endif
}

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
    int* validMask,
    float uvScale,
    bool isHalfAccumulation
)
{
    dim3 dimBlock(16, 16);
    dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) / dimBlock.y);
    inferInt8Syn<<<dimGrid, dimBlock>>>(
        weight,
        packedInput,
        quantizationScales,
        HP,
        DP,
        UP,
        TP,
        InvP,
        sampleList,
        output,
        width,
        height,
        validMask,
        uvScale,
        isHalfAccumulation
    );
}

__global__ void inferInt8(
    int* weight,
    int* packedInput,
    float* quantizationScales,
    cudaTextureObject_t HP,
    cudaTextureObject_t DP,
    cudaTextureObject_t UP,
    float* output,
    unsigned int width,
    unsigned int height,
    int* validMask,
    float uvScale,
    bool isHalfAccumulation
)
{
    __shared__ int W[768];
    unsigned int localIdx = threadIdx.y * blockDim.x + threadIdx.x;
    if (localIdx < 256)
    {
        W[3 * localIdx] = weight[3 * localIdx];
        W[3 * localIdx + 1] = weight[3 * localIdx + 1];
        W[3 * localIdx + 2] = weight[3 * localIdx + 2];
    }
    __syncthreads();

    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;
    if (validMask[y * width + x] == 0)
        return;

    int val1[HIDDEN_NUM];
    int val2[HIDDEN_PACKED_NUM];

    float h1, h2;
    float d1, d2;
    float u, v;
    unpackUnorm2x16(packedInput[4 * (y * width + x) + 0], h1, h2);
    unpackUnorm2x16(packedInput[4 * (y * width + x) + 1], d1, d2);
    unpackUnorm2x16(packedInput[4 * (y * width + x) + 2], u, v);

    float4 val = tex2DLayered<float4>(HP, h1, h2, 0);
    val2[0] = quantizeInt8x4f_safe(val, quantizationScales[0]);

    val = tex2DLayered<float4>(HP, h1, h2, 1);
    val2[1] = quantizeInt8x4f_safe(val, quantizationScales[0]);

    val = tex2DLayered<float4>(UP, v * uvScale, u * uvScale, 0);
    val2[2] = quantizeInt8x4f_safe(val, quantizationScales[0]);

    val = tex2DLayered<float4>(UP, v * uvScale, u * uvScale, 1);
    val2[3] = quantizeInt8x4f_safe(val, quantizationScales[0]);

    val = tex2DLayered<float4>(DP, d1, d2, 0);
    val2[4] = quantizeInt8x4f_safe(val, quantizationScales[0]);

    val = tex2DLayered<float4>(DP, d1, d2, 1);
    val2[5] = quantizeInt8x4f_safe(val, quantizationScales[0]);

    // layer 1
    for (int k = 0; k < HIDDEN_NUM; k++)
    {
        val1[k] = 0;
        for (int j = 0; j < IN_PACKED_NUM; j++)
        {
            val1[k] = __dp4a(val2[j], W[k * IN_PACKED_NUM + j], val1[k]);
        }
    }

    for (int k = 0; k < HIDDEN_PACKED_NUM; k++)
    {
        if (isHalfAccumulation)
            val2[k] = quantizeInt8x4h_safe(
                dequantizeInt8h_relu(val1[4 * k], quantizationScales[1]),
                dequantizeInt8h_relu(val1[4 * k + 1], quantizationScales[1]),
                dequantizeInt8h_relu(val1[4 * k + 2], quantizationScales[1]),
                dequantizeInt8h_relu(val1[4 * k + 3], quantizationScales[1]),
                quantizationScales[2]
            );

        else
            val2[k] = quantizeInt8x4f_safe(
                dequantizeInt8f_relu(val1[4 * k], quantizationScales[1]),
                dequantizeInt8f_relu(val1[4 * k + 1], quantizationScales[1]),
                dequantizeInt8f_relu(val1[4 * k + 2], quantizationScales[1]),
                dequantizeInt8f_relu(val1[4 * k + 3], quantizationScales[1]),
                quantizationScales[2]
            );
        // #endif
    }

    // layer 2
    for (int k = 0; k < HIDDEN_NUM; k++)
    {
        val1[k] = 0;
        for (int j = 0; j < HIDDEN_PACKED_NUM; j++)
        {
            val1[k] = __dp4a(val2[j], W[192 + k * HIDDEN_PACKED_NUM + j], val1[k]);
        }
    }
    for (int k = 0; k < HIDDEN_PACKED_NUM; k++)
    {
        if (isHalfAccumulation)
            val2[k] = quantizeInt8x4h_safe(
                dequantizeInt8h_relu(val1[4 * k], quantizationScales[3]),
                dequantizeInt8h_relu(val1[4 * k + 1], quantizationScales[3]),
                dequantizeInt8h_relu(val1[4 * k + 2], quantizationScales[3]),
                dequantizeInt8h_relu(val1[4 * k + 3], quantizationScales[3]),
                quantizationScales[4]
            );

        // #else
        else
            val2[k] = quantizeInt8x4f_safe(
                dequantizeInt8f_relu(val1[4 * k], quantizationScales[3]),
                dequantizeInt8f_relu(val1[4 * k + 1], quantizationScales[3]),
                dequantizeInt8f_relu(val1[4 * k + 2], quantizationScales[3]),
                dequantizeInt8f_relu(val1[4 * k + 3], quantizationScales[3]),
                quantizationScales[4]
            );
        // #endif
    }

    // layer 3
    for (int k = 0; k < HIDDEN_NUM; k++)
    {
        val1[k] = 0;
        for (int j = 0; j < HIDDEN_PACKED_NUM; j++)
        {
            val1[k] = __dp4a(val2[j], W[448 + k * HIDDEN_PACKED_NUM + j], val1[k]);
        }
    }
    for (int k = 0; k < HIDDEN_PACKED_NUM; k++)
    {
        if (isHalfAccumulation)
            val2[k] = quantizeInt8x4h_safe(
                dequantizeInt8h_relu(val1[4 * k], quantizationScales[5]),
                dequantizeInt8h_relu(val1[4 * k + 1], quantizationScales[5]),
                dequantizeInt8h_relu(val1[4 * k + 2], quantizationScales[5]),
                dequantizeInt8h_relu(val1[4 * k + 3], quantizationScales[5]),
                quantizationScales[6]
            );

        else
            val2[k] = quantizeInt8x4f_safe(
                dequantizeInt8f_relu(val1[4 * k], quantizationScales[5]),
                dequantizeInt8f_relu(val1[4 * k + 1], quantizationScales[5]),
                dequantizeInt8f_relu(val1[4 * k + 2], quantizationScales[5]),
                dequantizeInt8f_relu(val1[4 * k + 3], quantizationScales[5]),
                quantizationScales[6]
            );
    }

    // final layer
    for (int k = 0; k < 3; k++)
    {
        val1[k] = 0;
        for (int j = 0; j < HIDDEN_PACKED_NUM; j++)
        {
            val1[k] = __dp4a(val2[j], W[704 + k * HIDDEN_PACKED_NUM + j], val1[k]);
        }
    }
    __syncthreads();
    if (isHalfAccumulation)
    {
        output[4 * (y * width + x) + 0] = dequantizeInt8h_relu(val1[0], quantizationScales[7]);
        output[4 * (y * width + x) + 1] = dequantizeInt8h_relu(val1[1], quantizationScales[7]);
        output[4 * (y * width + x) + 2] = dequantizeInt8h_relu(val1[2], quantizationScales[7]);
    }

    else
    {
        output[4 * (y * width + x) + 0] = dequantizeInt8f_relu(val1[0], quantizationScales[7]);
        output[4 * (y * width + x) + 1] = dequantizeInt8f_relu(val1[1], quantizationScales[7]);
        output[4 * (y * width + x) + 2] = dequantizeInt8f_relu(val1[2], quantizationScales[7]);
    }
}

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
    int* validMask,
    float uvScale,
    bool isHalfAccumulation
)
{
    dim3 dimBlock(16, 16);
    dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) / dimBlock.y);
    inferInt8<<<dimGrid, dimBlock>>>(weight, packedInput, quantizationScales, HP, DP, UP, output, width, height, validMask, uvScale, isHalfAccumulation);
}
