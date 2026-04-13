/***************************************************************************
 # Copyright (c) 2015-24, NVIDIA CORPORATION. All rights reserved.
 #
 # Redistribution and use in source and binary forms, with or without
 # modification, are permitted provided that the following conditions
 # are met:
 #  * Redistributions of source code must retain the above copyright
 #    notice, this list of conditions and the following disclaimer.
 #  * Redistributions in binary form must reproduce the above copyright
 #    notice, this list of conditions and the following disclaimer in the
 #    documentation and/or other materials provided with the distribution.
 #  * Neither the name of NVIDIA CORPORATION nor the names of its
 #    contributors may be used to endorse or promote products derived
 #    from this software without specific prior written permission.
 #
 # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY
 # EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 # PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 # CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 # EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,w
 # PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 # PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 # OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 # (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **************************************************************************/

#include "Scene/SceneDefines.slangh"
#include "Utils/Math/MathConstants.slangh"
import Scene.Raytracing;
import Scene.Intersection;
import Utils.Math.MathHelpers;
import Utils.Geometry.IntersectionHelpers;
import Utils.Geometry.GeometryHelpers;
import Utils.Sampling.SampleGenerator;
import Utils.Math.FormatConversion;
import Utils.Debug.PixelDebug;
import Rendering.Lights.LightHelpers;
import Rendering.Lights.EnvMapSampler;

import Tools.BTFUtils;
import Utils.Neural.NBTF;
cbuffer CB
{
    uint gFrameCount;    // Frame count since scene was loaded.
    uint gPRNGDimension; // First available PRNG dimension.
    uint2 gRenderTargetDim;
    float4 gControlParas;

    bool gTracedShadowRay;
    bool gApplySyn;

    EnvMapSampler envMapSampler;
    NBTF gNBTF;
}

// Outputs
RWTexture2D<float4> gOutputColor;

SamplerState gSampler;
SamplerState gMaxSampler;

// Static configuration based on defines set from the host.
#define is_valid(name) (is_valid_##name != 0)
static const bool kUseEnvLight = USE_ENV_LIGHT;
static const float kRayTMax = FLT_MAX;

#define MAX_HEIGHT gControlParas.x
#define UV_SCALE gControlParas.y
#define HF_OFFSET gControlParas.z
#define HF_SCALE gControlParas.w
#define SHADOW_OFFSET 0.1

/**
 * Payload for shadow ray.
 */
struct ShadowRayData
{
    bool visible;
};

/**
 * Payload for scatter ray (up to 72B).
 */
struct ScatterRayData
{
    float3 radiance;  ///< Accumulated outgoing radiance from path.
    bool terminated;  ///< Set to true when path is terminated.
    float3 origin;    ///< Next path segment origin.
    float3 direction; ///< Next path segment direction.

    SampleGenerator sg; ///< Per-ray state for the sample generator (up to 16B).

    /**
     * Initializes ray payload with default parameters.
     */
    __init(SampleGenerator sg)
    {
        this.terminated = false;
        this.radiance = float3(0, 0, 0);
        this.origin = float3(0, 0, 0);
        this.direction = float3(0, 0, 0);
        this.sg = sg;
    }
};

/**
 * Setup ShadingData based on loaded vertex/material attributes for a hit point.
 * @param[in] hit Hit information.
 * @param[in] rayOrigin Ray origin.
 * @param[in] rayDir Normalized ray direction.
 * @param[in] lod Method for computing texture level-of-detail.
 * @return ShadingData struct.
 */
ShadingData loadShadingData(const HitInfo hit, const float3 rayOrigin, const float3 rayDir)
{
    VertexData v = {};
    uint materialID = {};

#if SCENE_HAS_GEOMETRY_TYPE(GEOMETRY_TYPE_TRIANGLE_MESH)
    if (hit.getType() == HitType::Triangle)
    {
        const TriangleHit triangleHit = hit.getTriangleHit();
        v = gScene.getVertexData(triangleHit);
        materialID = gScene.getMaterialID(triangleHit.instanceID);
    }
#endif
    ShadingData sd = gScene.materials.prepareShadingData(v, materialID, -rayDir);

    return sd;
}

ShadingData loadShadingData(
    const HitInfo hit,
    const float3 rayOrigin,
    const float3 rayDir,
    out GeometryInstanceID instanceID,
    uint primitiveIndex
)
{
    VertexData v = {};
    uint materialID = {};

    if (hit.getType() == HitType::DisplacedTriangle)
    {
        const DisplacedTriangleHit displacedTriangleHit = hit.getDisplacedTriangleHit();
        instanceID = displacedTriangleHit.instanceID;
        primitiveIndex = displacedTriangleHit.primitiveIndex;
        v = gScene.getVertexData(displacedTriangleHit, -rayDir);
        materialID = gScene.getMaterialID(displacedTriangleHit.instanceID);
    }
    ShadingData sd = gScene.materials.prepareShadingData(v, materialID, -rayDir);

    return sd;
}
/**
 * Returns the primary ray's direction.
 */
float3 getPrimaryRayDir(uint2 launchIndex, uint2 launchDim, const Camera camera, float2 jitter)
{
    // Compute sample position in screen space in [0,1] with origin at the top-left corner.
    // The camera jitter offsets the sample by +-0.5 pixels from the pixel center.
    float2 p = (launchIndex + jitter) / launchDim;
    float2 ndc = float2(2, -2) * p + float2(-1, 1);
    // Compute the non-normalized ray direction assuming a pinhole camera.
    return normalize(ndc.x * camera.data.cameraU + ndc.y * camera.data.cameraV + camera.data.cameraW);
}

/**
 * Traces a shadow ray towards a light source.
 * @param[in] origin Ray origin for the shadow ray.
 * @param[in] dir Direction from shading point towards the light source (normalized).
 * @param[in] distance Distance to the light source.
 * @return True if light is visible, false otherwise.
 */
bool traceShadowRay(float3 origin, float3 dir, float distance)
{
    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = dir;
    ray.TMin = 0.f;
    ray.TMax = distance;

    ShadowRayData rayData;
    rayData.visible = false; // Set to true by miss shader if ray is not terminated before
    TraceRay(
        gScene.rtAccel,
        RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH,
        0xff /* instanceInclusionMask */,
        1 /* hitIdx */,
        rayTypeCount,
        1 /* missIdx */,
        ray,
        rayData
    );

    return rayData.visible;
}

/**
 * Traces a scatter ray based on ray parameters stored in the ray payload.
 * @param[in] rayData Describes the ray parameters. The struct is modified based on the result.
 */
void traceScatterRay(inout ScatterRayData rayData)
{
    RayDesc ray;
    ray.Origin = rayData.origin;
    ray.Direction = rayData.direction;
    ray.TMin = 0.f;
    ray.TMax = kRayTMax;

    uint rayFlags = 0; // TODO: Set cull mode from the app
    TraceRay(gScene.rtAccel, rayFlags, 0xff /* instanceInclusionMask */, 0 /* hitIdx */, rayTypeCount, 0 /* missIdx */, ray, rayData);
}

// Envmap sampling
bool generateEnvMapSample(float3 origin, inout SampleGenerator sg, inout float3 Li, inout float3 dir, inout float pdf)
{
    if (!kUseEnvLight)
        return false;

    EnvMapSample lightSample;
    if (!envMapSampler.sample(sampleNext2D(sg), lightSample))
        return false;

    bool V = traceShadowRay(origin, lightSample.dir, kRayTMax);
    if (!V)
        return false;

    Li = lightSample.pdf > 0.f ? lightSample.Le / lightSample.pdf : float3(0);
    pdf = lightSample.pdf;
    dir = lightSample.dir;

    return any(Li > 0.f);
}

void computeRayDifferentials(
    const TriangleHit hit,
    float3 rayDir,
    float hitT,
    const Camera camera,
    float2 invFrameDim,
    out float2 ddx,
    out float2 ddy
)
{
    // TODO: Is this code correct for instance transforms that flip the handedness of the coordinate system?

    // Ray differentials
    float3 P[3];
    gScene.getVertexPositionsW(hit.instanceID, hit.primitiveIndex, P);
    float3 e1 = P[1] - P[0];
    float3 e2 = P[2] - P[0];
    float3 d = rayDir;
    float k = dot(cross(e1, e2), d);
    k = abs(k) > 1e-20f ? rcp(k) : 0.0f;
    float3 cu = cross(e2, d);
    float3 cv = cross(d, e1);
    // Assumes a normalized ray direction
    // dDdx in ray gen
    float3 dx = camera.data.cameraU * 2.f * invFrameDim.x / camera.data.focalDistance;
    // dDdy in ray gen
    float3 dy = camera.data.cameraV * 2.f * invFrameDim.y / camera.data.focalDistance;
    // Transfer to primary hit
    float3 q = dx * hitT;
    float3 r = dy * hitT;
    float dudx = k * dot(cu, q);
    float dudy = k * dot(cu, r);
    float dvdx = k * dot(cv, q);
    float dvdy = k * dot(cv, r);
    float2 T[3];
    gScene.getVertexTexCoords(hit.instanceID, hit.primitiveIndex, T);
    float2 g1 = T[1] - T[0];
    float2 g2 = T[2] - T[0];
    float dsdx = (dudx * g1.x + dvdx * g2.x);
    float dsdy = (dudy * g1.x + dvdy * g2.x);
    float dtdx = (dudx * g1.y + dvdx * g2.y);
    float dtdy = (dudy * g1.y + dvdy * g2.y);
    ddx = float2(dsdx, dtdx);
    ddy = float2(dsdy, dtdy);
}

float4 tracePath(const uint2 pixel, const uint2 frameDim)
{
    // Create sample generator.
    SampleGenerator sg = SampleGenerator(pixel, gFrameCount);

    // Advance the generator to the first available dimension.
    // TODO: This is potentially expensive. We may want to store/restore the state from memory if it becomes a problem.
    for (uint i = 0; i < gPRNGDimension; i++)
        sampleNext1D(sg);
    const float3 primaryRayOrigin = gScene.camera.getPosition();
    const float3 primaryRayDir = getPrimaryRayDir(pixel, frameDim, gScene.camera, sampleNext2D(sg));
    // Prepare ray payload.
    ScatterRayData rayData = ScatterRayData(sg);
    rayData.origin = primaryRayOrigin;
    rayData.direction = primaryRayDir;

    // ===========================  NeuMat  ===========================
    // Only the valid pixels are needed for cuda inference.
    traceScatterRay(rayData);
    return float4(rayData.radiance, 1.0f);
}

//
// Shader entry points for miss shaders.
//
[shader("miss")]
void scatterMiss(inout ScatterRayData rayData)
{
    // Ray missed the scene. Mark the ray as terminated.
    rayData.terminated = true;

    uint2 pixel = DispatchRaysIndex().xy;

    // Add contribution from distant light (env map) in this direction.
    if (kUseEnvLight)
    {
        float3 Le = gScene.envMap.eval(WorldRayDirection());
        rayData.radiance += Le;
    }
}

[shader("miss")]
void shadowMiss(inout ShadowRayData rayData)
{
    // The miss shader is executed if the ray misses all geometry. Mark as visible.
    rayData.visible = true;
}

//
// Shader entry points for TriangleMesh hit groups.
//
[shader("anyhit")]
void scatterTriangleMeshAnyHit(inout ScatterRayData rayData, BuiltInTriangleIntersectionAttributes attribs)
{
    // Alpha test for non-opaque geometry.
    GeometryInstanceID instanceID = getGeometryInstanceID();
    VertexData v = getVertexData(instanceID, PrimitiveIndex(), attribs);
    const uint materialID = gScene.getMaterialID(instanceID);
    if (gScene.materials.alphaTest(v, materialID, 0.f))
        IgnoreHit();
}

[shader("closesthit")]
void scatterTriangleMeshClosestHit(inout ScatterRayData rayData, BuiltInTriangleIntersectionAttributes attribs)
{
    TriangleHit triangleHit;
    triangleHit.instanceID = getGeometryInstanceID();
    triangleHit.primitiveIndex = PrimitiveIndex();
    triangleHit.barycentrics = attribs.barycentrics;
    handleHit(HitInfo(triangleHit), rayData);
}

[shader("anyhit")]
void shadowTriangleMeshAnyHit(inout ShadowRayData rayData, BuiltInTriangleIntersectionAttributes attribs)
{
    // Alpha test for non-opaque geometry.
    GeometryInstanceID instanceID = getGeometryInstanceID();
    VertexData v = getVertexData(instanceID, PrimitiveIndex(), attribs);
    const uint materialID = gScene.getMaterialID(instanceID);
    if (gScene.materials.alphaTest(v, materialID, 0.f))
        IgnoreHit();
}

//
// Shader entry point for ray generation shader.
//
[shader("raygeneration")]
void rayGen()
{
    uint2 pixel = DispatchRaysIndex().xy;
    uint2 frameDim = DispatchRaysDimensions().xy;
    gOutputColor[pixel] = tracePath(pixel, frameDim);
}

// hit non-displaced triangle
void handleHit(const HitInfo hit, inout ScatterRayData rayData)
{
    // Load shading data.
    ShadingData sd = loadShadingData(hit, rayData.origin, rayData.direction);

    // Create material instance.
    let lod = ExplicitLodTextureSampler(0.f);
    let mi = gScene.materials.getMaterialInstance(sd, lod);

    float3 wo = sd.frame.toLocal(sd.V);
    float3 wiWS; // world space incident direction
    float3 envLi;
    float envPdf;
    // We only use the envmap as light source
    if (generateEnvMapSample(sd.computeRayOrigin(), rayData.sg, envLi, wiWS, envPdf))
    {
        if (sd.materialID == 1)
        {
            float2 ddx, ddy;
            float hitT = length(sd.posW - gScene.camera.getPosition());
            computeRayDifferentials(hit.getTriangleHit(), -sd.V, hitT, gScene.camera, 1.0f / float2(gRenderTargetDim), ddx, ddy);
            float3 wi = sd.frame.toLocal(wiWS);
            float NdotL = wi.z > 0 ? wi.z : 0; // BTF already encodes the cosine term.
            uint4 packedData;
            uint2 pixel = DispatchRaysIndex().xy;
            rayData.radiance = gNBTF.evalDebug(wi, wo, frac(abs(sd.uv)) * UV_SCALE, ddx, ddy, gApplySyn).rgb * envLi * NdotL;
            rayData.terminated = true;
        }
        else
        {
            rayData.terminated = true;
            rayData.radiance += envLi * mi.eval(sd, wiWS, rayData.sg);
        }
    }
}

