#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "CUDAConstant.h"


// =====================================================================================================================
// Activation Functions
// =====================================================================================================================
float __device__ __forceinline__ relu(float x)
{
    return max(x, 0.0f);
}
__half __device__ __forceinline__ relu(__half x)
{
    return __hmax(x, CUDART_ZERO_FP16);
}
int __device__ __forceinline__ relu(int x)
{
    return max(x, 0);
}
float __device__ __forceinline__ leakyrelu(float x)
{
    return max(x, 0.0f) + min(x, 0.0f) * 0.01f;
}
__half __device__ __forceinline__ leakyrelu(__half x)
{
    return __hmax(x, CUDART_ZERO_FP16) + __hmul(__hmin(x, CUDART_ZERO_FP16), __float2half_rn(0.01f));
}




// =====================================================================================================================
// Packing and Unpacking Functions
// =====================================================================================================================
inline __device__ void unpackSnorm2x16(unsigned int packed, float& a, float& b)
{
    a = __int2float_rd((int)(packed << 16) >> 16) / 32767.f;
    b = __int2float_rd((int)packed >> 16) / 32767.f;
}

inline __device__ void unpackUnorm2x16(unsigned int packed, float& a, float& b)
{
    a = __uint2float_rd((unsigned int)(packed << 16) >> 16) / 65535.f;
    b = __uint2float_rd((unsigned int)packed >> 16) / 65535.f;
}


inline __device__ void unpackSnorm2x16(unsigned int packed, __half& a, __half& b)
{
    a = __hdiv(__int2half_rd((int)(packed << 16) >> 16), 32767);
    b = __hdiv(__int2half_rd((int)packed >> 16), 32767);
}

inline __device__ void unpackSnorm2x16(int packed, __half& a, __half& b)
{
    a = __hdiv(__int2half_rd((int)(packed << 16) >> 16), 32767);
    b = __hdiv(__int2half_rd((int)packed >> 16), 32767);
}


inline __device__ short2 packInt2x16(int a)
{
    return make_short2((short)((a << 16) >> 16), (short)(a >> 16));
}

inline __device__ int clampInt8(int a)
{
    return min(127, max(-127, a));
}
inline __device__ int quantizeInt8x4f_safe(float a, float b, float c, float d, const float scale)
{
    return (clampInt8(__float2int_rn((a * scale))) & 0x000000ff) | (clampInt8(__float2int_rn(b * scale)) << 8) & 0x0000ff00 |
           (clampInt8(__float2int_rn(c * scale)) << 16) & 0x00ff0000 | (clampInt8(__float2int_rn(d * scale)) << 24) & 0xff000000;
}
inline __device__ int quantizeInt8x4f_safe(float4 v, const float scale)
{
    return (clampInt8(__float2int_rn((v.x * scale))) & 0x000000ff) | (clampInt8(__float2int_rn(v.y * scale)) << 8) & 0x0000ff00 |
           (clampInt8(__float2int_rn(v.z * scale)) << 16) & 0x00ff0000 | (clampInt8(__float2int_rn(v.w * scale)) << 24) & 0xff000000;
}

inline __device__ int quantizeInt8x4h_safe(__half a, __half b, __half c, __half d, const __half scale)
{
    return (clampInt8(__half2int_rn(__hmul(a, scale))) & 0x000000ff) | (clampInt8(__half2int_rn(__hmul(b, scale))) << 8) & 0x0000ff00 |
           (clampInt8(__half2int_rn(__hmul(c, scale))) << 16) & 0x00ff0000 |
           (clampInt8(__half2int_rn(__hmul(d, scale))) << 24) & 0xff000000;
}

__forceinline__ __device__ float dequantizeInt8(const int packedData, const float scale)
{
    return __int2float_rn(packedData) * scale;
}

__forceinline__ __device__ float dequantizeInt8f_relu(const int packedData, const float scale)
{
    // return relu(__int2float_rn(packedData) * scale);
    return __int2float_rn(relu(packedData)) * scale;
}
__forceinline__ __device__ __half dequantizeInt8h_relu(const int packedData, const __half scale)
{
    // return relu(__hmul(__int2half_rn(packedData), scale));
    return __hmul(__int2half_rn(relu(packedData)), scale);
}
__forceinline__ __device__ void dequantizeInt8x4(const int packedData, __half& a, __half& b, __half& c, __half& d, const __half scale)
{
    a = __hmul(__int2half_rn((int)packedData << 24 >> 24), scale);
    b = __hmul(__int2half_rn((int)packedData << 16 >> 24), scale);
    c = __hmul(__int2half_rn((int)packedData << 8 >> 24), scale);
    d = __hmul(__int2half_rn((int)packedData >> 24), scale);
}
__forceinline__ __device__ void dequantizeInt8x4(const int packedData, float& a, float& b, float& c, float& d, const float scale)
{
    a = __int2float_rn((int)packedData << 24 >> 24) * scale;
    b = __int2float_rn((int)packedData << 16 >> 24) * scale;
    c = __int2float_rn((int)packedData << 8 >> 24) * scale;
    d = __int2float_rn((int)packedData >> 24) * scale;
}
__forceinline__ __device__ void unpackInt8x4(const int packedData, int& a, int& b, int& c, int& d)
{
    a = (int)packedData << 24 >> 24;
    b = (int)packedData << 16 >> 24;
    c = (int)packedData << 8 >> 24;
    d = (int)packedData >> 24;
}
__forceinline__ __device__ int packInt16x2(int a, int b)
{
    return (a & 0x0000ffff) | ((b << 16) & 0xffff0000);
}


// =====================================================================================================================
// Synthesis Functions
// =====================================================================================================================
__forceinline__ __device__ float clampG(float x) {
    return fminf(fmaxf(x, 0.0001), 0.999);
}
inline __device__ float rnd21(float2 p)
{
    float temp = sinf(12.9898f * p.x + 78.233f * p.y) * 43758.5453f;
    return (temp - floor(temp));
}

__forceinline__  __device__ float rnd21(float p1, float p2)
{
    float temp = sinf(12.9898f * p1 + 78.233f * p2) * 43758.5453f;
    return (temp - floor(temp));
}
inline __device__ float B0cos(float2 uv)
{
    float cosu = sinf(uv.x * 3.14159265f);
    float cosv = sinf(uv.y * 3.14159265f);
    return powf(cosu * cosv * cosu * cosv, 0.5f);
}


__forceinline__  __device__ float B0cos(float u, float v)
{
    float cosu = sinf(u * 3.14159265f);
    float cosv = sinf(v * 3.14159265f);
    return powf(cosu * cosv * cosu * cosv, 0.5f);
}

__forceinline__  __device__ float B1cos(float2 uv)
{
    uv = float2{uv.x + 0.5f, uv.y + 0.5f};
    float cosu = sinf(uv.x * 3.14159265f);
    float cosv = sinf(uv.y * 3.14159265f);
    return powf(cosu * cosv * cosu * cosv, 0.5f);
}

inline __device__ float B1cos(float u, float v)
{
    float cosu = sinf((u + 0.5f)* 3.14159265f);
    float cosv = sinf((v + 0.5f) * 3.14159265f);
    return powf(cosu * cosv * cosu * cosv, 0.5f);
}

inline __device__ float BSingularity(float2 uv)
{
    uv = float2{(uv.x - 0.5f) * 1.41421356237f, (uv.y - 0.5f) * 1.41421356237f};
    const float a = 0.78539816f; // Pi / 4
    float cosA = cosf(a);
    float sinA = sinf(a);
    float2 V = float2{cosA * uv.x + sinA * uv.y, -sinA * uv.x + cosA * uv.y};
    float cosu = sinf(V.x * 3.14159265f);
    float cosv = sinf(V.y * 3.14159265f);
    return 0.02f * cosu * cosv * cosu * cosv;
}
inline __device__ float BSingularity(float u, float v)
{
    u = (u - 0.5f) * 1.41421356237f;
    v = (v - 0.5f) * 1.41421356237f;
    const float a = 0.78539816f; // Pi / 4
    float cosA = cosf(a);
    float sinA = sinf(a);
    float newU = cosA * u + sinA * v;
    float newV = -sinA * u + cosA * v;
    float cosu = sinf(newU * 3.14159265f);
    float cosv = sinf(newV * 3.14159265f);
    return 0.02f * cosu * cosv * cosu * cosv;
}

__device__ void TriangleGrid(float& w1, float& w2, float& w3, int2& vertex1, int2& vertex2, int2& vertex3, float2 st)
{
    st = float2{3.4641016f * st.x, 3.4641016f * st.y};
    float2 skewedCoord = float2{st.x - 0.57735027f * st.y, 1.15470054f * st.y};
    int2 baseId = int2{(int)floor(skewedCoord.x), (int)floor(skewedCoord.y)};
    float2 fracPart = float2{abs(skewedCoord.x - trunc(skewedCoord.x)), abs(skewedCoord.y - trunc(skewedCoord.y))};
    float3 temp = float3{fracPart.x, fracPart.y, 1.0f - fracPart.x - fracPart.y};
    if (temp.z > 0)
    {
        w1 = temp.z;
        w2 = temp.y;
        w3 = temp.x;
        vertex1 = int2{baseId.x, baseId.y};
        vertex2 = int2{baseId.x, baseId.y + 1};
        vertex2 = int2{baseId.x + 1, baseId.y};
    }
    else
    {
        w1 = -temp.z;
        w2 = 1.0f - temp.y;
        w3 = 1.0f - temp.x;
        vertex1 = int2{baseId.x + 1, baseId.y + 1};
        vertex2 = int2{baseId.x + 1, baseId.y};
        vertex2 = int2{baseId.x, baseId.y + 1};
    }
}
inline __device__ float2 hash22(float2 p)
{
    float2 r = float2{127.1f * p.x + 311.7f * p.y, 269.5f * p.x + 183.3f * p.y};
    //float2 temp = float2{sinf(r.x) * 43758.5453f, sinf(r.y) * 43758.5453f};
    float2 temp = float2{sinf(r.x), sinf(r.y)};
    //return float2{temp.x - floor(temp.x), temp.y - floor(temp.y)};
    return temp;
}
