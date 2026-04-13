#include "Utils/Math/MathConstants.slangh"

import MLP;
import Utils.Math.MathHelpers;
import Utils.Texture.HistogramBlending;
import Utils.Math.FormatConversion;
#ifndef MAX_NN_DIM
#define MAX_NN_DIM 8
#endif

struct NBTF
{
    Texture2DArray<float4> uP;
    Texture2DArray<float4> hP;
    Texture2DArray<float4> dP;

    HistogramBlendingFeatureData histoFeatureData;

    int2 uDims;
    int2 hDims;
    int2 dDims;

    int inputSize;
    MLP mlp;

    SamplerState gSampler;


    float3 evalDebug(float2 h, float2 d, float2 uv, float2 ddx, float2 ddy, bool synthesize = false)
    {

        float4 x[MAX_NN_DIM];
        int count = 0;
        for (uint i = 0; i < hDims.y; i++)
        {
            x[count] = hP.SampleLevel(gSampler, float3(h, i), 0.f);
            count += 1;
        }
        for (uint i = 0; i < uDims.y; i++)
        {
            if (synthesize)
                // return histoFeatureData.acf.SampleLevel(gSampler, uv, 0);
                x[count] = HistorgramBlending(histoFeatureData, gSampler, uv, i, ddx, ddy);
            // x[count] = HexTiling(uP, gSampler, ddx, ddy, uv, i);
            else
            {
                x[count] = uP.SampleGrad(gSampler, float3(uv, i), ddx, ddy);
            }

            count += 1;
        }

        for (uint i = 0; i < dDims.y; i++)
        {
            x[count] = dP.SampleLevel(gSampler, float3(d, i), 0.f);
            count += 1;
        }
        return mlp.eval(x).xyz;
    }
    float4 evalDebugFP32(float2 h, float2 d, float2 uv, float2 ddx, float2 ddy, bool synthesize = false)
    {
        float x[24];
        int count = 0;
        for (uint i = 0; i < hDims.y; i++)
        {
            float4 val = hP.SampleLevel(gSampler, float3(h, i), 0.f);
            x[count * 4] = val.x;
            x[count * 4 + 1] = val.y;
            x[count * 4 + 2] = val.z;
            x[count * 4 + 3] = val.w;
            count += 1;
        }
        for (uint i = 0; i < uDims.y; i++)
        {
            float4 val;
            if (synthesize)
                val = HistorgramBlending(histoFeatureData, gSampler, uv, i, ddx, ddy);
            else
                val = uP.SampleGrad(gSampler, float3(uv, i), ddx, ddy);
            x[count * 4] = val.x;
            x[count * 4 + 1] = val.y;
            x[count * 4 + 2] = val.z;
            x[count * 4 + 3] = val.w;
            count += 1;
        }

        for (uint i = 0; i < dDims.y; i++)
        {
            float4 val = dP.SampleLevel(gSampler, float3(d, i), 0.f);
            x[count * 4] = val.x;
            x[count * 4 + 1] = val.y;
            x[count * 4 + 2] = val.z;
            x[count * 4 + 3] = val.w;
            count += 1;
        }
        return mlp.eval(x);
    }







    float3 evalDebug(float3 wi, float3 wo, float2 uv, float2 ddx, float2 ddy, bool synthesize = false)
    {
        float2 h, d;
        dirToRusinkiewicz(wi, wo, h, d);
        float3 vh = spherical_to_cartesian_rad(h);
        float3 vd = spherical_to_cartesian_rad(d);

        h = hemisphereDirToDiskSquare(vh);
        d = hemisphereDirToDiskSquare(vd);

        float4 x[MAX_NN_DIM];
        int count = 0;
        for (uint i = 0; i < hDims.y; i++)
        {
            x[count] = hP.SampleLevel(gSampler, float3(h, i), 0.f);
            count += 1;
        }
        for (uint i = 0; i < uDims.y; i++)
        {
            if (synthesize)
                // return histoFeatureData.acf.SampleLevel(gSampler, uv, 0);
                x[count] = HistorgramBlending(histoFeatureData, gSampler, uv, i, ddx, ddy);
            // x[count] = HexTiling(uP, gSampler, ddx, ddy, uv, i);
            else
            {
                x[count] = uP.SampleGrad(gSampler, float3(uv, i), ddx, ddy);
            }

            count += 1;
        }

        for (uint i = 0; i < dDims.y; i++)
        {
            x[count] = dP.SampleLevel(gSampler, float3(d, i), 0.f);
            count += 1;
        }
        return mlp.eval(x).xyz;
    }
    float4 evalDebugFP32(float3 wi, float3 wo, float2 uv, float2 ddx, float2 ddy, bool synthesize = false)
    {
        float2 h, d;
        dirToRusinkiewicz(wi, wo, h, d);
        float3 vh = spherical_to_cartesian_rad(h);
        float3 vd = spherical_to_cartesian_rad(d);

        h = hemisphereDirToDiskSquare(vh);
        d = hemisphereDirToDiskSquare(vd);

        float x[24];
        int count = 0;
        for (uint i = 0; i < hDims.y; i++)
        {
            float4 val = hP.SampleLevel(gSampler, float3(h, i), 0.f);
            x[count * 4] = val.x;
            x[count * 4 + 1] = val.y;
            x[count * 4 + 2] = val.z;
            x[count * 4 + 3] = val.w;
            count += 1;
        }
        for (uint i = 0; i < uDims.y; i++)
        {
            float4 val;
            if (synthesize)
                val = HistorgramBlending(histoFeatureData, gSampler, uv, i, ddx, ddy);
            else
                val = uP.SampleGrad(gSampler, float3(uv, i), ddx, ddy);
            x[count * 4] = val.x;
            x[count * 4 + 1] = val.y;
            x[count * 4 + 2] = val.z;
            x[count * 4 + 3] = val.w;
            count += 1;
        }

        for (uint i = 0; i < dDims.y; i++)
        {
            float4 val = dP.SampleLevel(gSampler, float3(d, i), 0.f);
            x[count * 4] = val.x;
            x[count * 4 + 1] = val.y;
            x[count * 4 + 2] = val.z;
            x[count * 4 + 3] = val.w;
            count += 1;
        }
        return mlp.eval(x);
    }
};