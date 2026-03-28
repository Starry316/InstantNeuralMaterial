import Utils.Debug.PixelDebug;
cbuffer PerFrameCB
{
    uint2 gRenderTargetDim;
    bool gNeedHDRRecon;
}
// final render image
RWTexture2D<float4> gOutputColor;
// reflectance
Buffer<float> gInputColor;
Buffer<int> cudaValidBuffer;

[numthreads(16, 16, 1)]
void csMain(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    if (any(dispatchThreadId.xy >= gRenderTargetDim))
        return;
    uint2 pixel = dispatchThreadId.xy;

    if (cudaValidBuffer[pixel.y * gRenderTargetDim.x + pixel.x] == 0)
        return;
    float4 btfVal = float4(
        gInputColor[4 * (pixel.y * gRenderTargetDim.x + pixel.x) + 0],
        gInputColor[4 * (pixel.y * gRenderTargetDim.x + pixel.x) + 1],
        gInputColor[4 * (pixel.y * gRenderTargetDim.x + pixel.x) + 2],
        gInputColor[4 * (pixel.y * gRenderTargetDim.x + pixel.x) + 3]
    );


    if (any(isnan(btfVal)) || any(isinf(btfVal)))
        btfVal = float4(0);
    if (gNeedHDRRecon) {
        btfVal = exp(btfVal) - 1.0f;
        // btfVal = pow(btfVal, 1.0f / 1.4f);
    }

    gOutputColor[pixel] *= btfVal;
}
