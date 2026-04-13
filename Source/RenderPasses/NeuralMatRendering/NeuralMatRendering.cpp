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
 # EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 # PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 # PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 # OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 # (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **************************************************************************/
#include <fstream>
#include "NeuralMatRendering.h"
#include "RenderGraph/RenderPassHelpers.h"
#include "RenderGraph/RenderPassStandardFlags.h"
#include "Utils/CudaUtils.h"
#include "Utils/Neural/IOHelper.h"
#include "Tools/Utils.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define MAX_HEIGHT mControlParas.x
#define UV_SCALE mControlParas.y
#define HF_OFFSET mControlParas.z
#define HF_SCALE mControlParas.w

extern "C" FALCOR_API_EXPORT void registerPlugin(Falcor::PluginRegistry& registry)
{
    registry.registerClass<RenderPass, NeuralMatRendering>();
}

namespace
{

const char kShaderFile[] = "RenderPasses/NeuralMatRendering/MinimalPathTracer.rt.slang";

// Ray tracing settings that affect the traversal stack size.
// These should be set as small as possible.
const uint32_t kMaxPayloadSizeBytes = 56u;
const uint32_t kMaxRecursionDepth = 2u;

const ChannelList kOutputChannels = {
    // clang-format off
    { "color",          "gOutputColor", "Output color (sum of direct and indirect)", false, ResourceFormat::RGBA32Float },
    // clang-format on
};

} // namespace

NeuralMatRendering::NeuralMatRendering(ref<Device> pDevice, const Properties& props) : RenderPass(pDevice)
{
    // Create a sample generator.
    mpSampleGenerator = SampleGenerator::create(mpDevice, SAMPLE_GENERATOR_UNIFORM);
    mpPixelDebug = std::make_unique<PixelDebug>(mpDevice);
    FALCOR_ASSERT(mpSampleGenerator);
}

Properties NeuralMatRendering::getProperties() const
{
    Properties props;
    return props;
}

RenderPassReflection NeuralMatRendering::reflect(const CompileData& compileData)
{
    RenderPassReflection reflector;

    // Define our input/output channels.
    // addRenderPassInputs(reflector, kInputChannels);
    addRenderPassOutputs(reflector, kOutputChannels);
    return reflector;
}

// This pass is used to get the primary ray's hit, and pack the input data for the neural network inference.
void NeuralMatRendering::tracingPass(RenderContext* pRenderContext, const RenderData& renderData)
{
    auto& dict = renderData.getDictionary();
    // Get dimensions of ray dispatch.
    const Falcor::uint2 targetDim = renderData.getDefaultTextureDims();
    FALCOR_ASSERT(targetDim.x > 0 && targetDim.y > 0);

    createBuffer(mpValidBuffer, mpDevice, targetDim, 1);
    // createBuffer(mpPackedInputBuffer, mpDevice, targetDim, 5);

    // Request the light collection if emissive lights are enabled.
    if (mpScene->getRenderSettings().useEmissiveLights)
    {
        mpScene->getLightCollection(pRenderContext);
    }

    // Specialize program.
    // These defines should not modify the program vars. Do not trigger program vars re-creation.
    mTracer.pProgram->addDefine("USE_ENV_LIGHT", mpScene->useEnvLight() ? "1" : "0");

    // For optional I/O resources, set 'is_valid_<name>' defines to inform the program of which ones it can access.
    // TODO: This should be moved to a more general mechanism using Slang.
    // mTracer.pProgram->addDefines(getValidResourceDefines(kInputChannels, renderData));
    mTracer.pProgram->addDefines(getValidResourceDefines(kOutputChannels, renderData));

    // Prepare program vars. This may trigger shader compilation.
    // The program should have all necessary defines set at this point.
    if (!mTracer.pVars)
        prepareVars();
    FALCOR_ASSERT(mTracer.pVars);

    // Set constants.
    auto var = mTracer.pVars->getRootVar();
    var["CB"]["gFrameCount"] = mFrameCount;
    var["CB"]["gPRNGDimension"] = dict.keyExists(kRenderPassPRNGDimension) ? dict[kRenderPassPRNGDimension] : 0u;
    var["CB"]["gControlParas"] = mControlParas;
    // var["CB"]["gCurvatureParas"] = mCurvatureParas;
    var["CB"]["gApplySyn"] = mApplySyn;
    var["CB"]["gRenderTargetDim"] = targetDim;

    mpNBTFFP32->bindShaderData(var["CB"]["gNBTF"]);


    if (mpEnvMapSampler)
        mpEnvMapSampler->bindShaderData(var["CB"]["envMapSampler"]);

    // Bind I/O buffers. These needs to be done per-frame as the buffers may change anytime.
    auto bind = [&](const ChannelDesc& desc)
    {
        if (!desc.texname.empty())
        {
            var[desc.texname] = renderData.getTexture(desc.name);
        }
    };

    for (auto channel : kOutputChannels)
        bind(channel);

    // Bind textures
    var["gOutputColor"] = renderData.getTexture("color");
    // var["gMaxSampler"] = mpMaxSampler;

    mpScene->raytrace(pRenderContext, mTracer.pProgram.get(), mTracer.pVars, Falcor::uint3(targetDim, 1));
    pRenderContext->submit(false);
    pRenderContext->signal(mpFence.get());
    mpFence->wait();
}


void NeuralMatRendering::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    // If we have no scene, just clear the outputs and return.
    if (!mpScene)
    {
        for (auto it : kOutputChannels)
        {
            Texture* pDst = renderData.getTexture(it.name).get();
            if (pDst)
                pRenderContext->clearTexture(pDst);
        }
        return;
    }
    // Update refresh flag if options that affect the output have changed.
    auto& dict = renderData.getDictionary();
    if (mOptionsChanged)
    {
        auto flags = dict.getValue(kRenderPassRefreshFlags, RenderPassRefreshFlags::None);
        dict[Falcor::kRenderPassRefreshFlags] = flags | Falcor::RenderPassRefreshFlags::RenderOptionsChanged;
        mOptionsChanged = false;
    }

    tracingPass(pRenderContext, renderData);
    mFrameCount++;
}

void NeuralMatRendering::renderUI(Gui::Widgets& widget)
{
    bool dirty = false;
   
    widget.dropdown(" ", mModelName);
    if (widget.button("Load model", true))
    {
        loadNetwork(mpDevice->getRenderContext());
        dirty = true;
    }

    dirty |= widget.slider("UV Scaling", UV_SCALE, 0.0f, 50.0f);
    dirty |= widget.checkbox("Apply Synthesis", mApplySyn);

    // if (auto group = widget.group("Envmap Control", false))
    {
        dirty |= widget.slider("Env rot X", mEnvRotAngle.x, 0.0f, 360.0f);
        if (widget.button("X -", true))
        {
            mEnvRotAngle.x -= 5;
            dirty = true;
        }
        if (widget.button("X +", true))
        {
            mEnvRotAngle.x += 5;
            dirty = true;
        }
        dirty |= widget.slider("Env rot Y", mEnvRotAngle.y, 0.0f, 360.0f);
        if (widget.button("Y -", true))
        {
            mEnvRotAngle.y -= 5;
            dirty = true;
        }
        if (widget.button("Y +", true))
        {
            mEnvRotAngle.y += 5;
            dirty = true;
        }
        dirty |= widget.slider("Env rot Z", mEnvRotAngle.z, 0.0f, 360.0f);
        if (widget.button("Z -", true))
        {
            mEnvRotAngle.z -= 5;
            dirty = true;
        }
        if (widget.button("Z +", true))
        {
            mEnvRotAngle.z += 5;
            dirty = true;
        }
    }
    

  

    if (widget.button("Reset Envmap (click this after loading a new scene)"))
    {
        mpEnvMapSampler = std::make_unique<EnvMapSampler>(mpDevice, mpScene->getEnvMap());
        dirty = true;
    }
    widget.tooltip("Refresh the importance sampling map for new loaded envmap", true);
    if (mpScene->getEnvMap())
    {
        mpScene->getEnvMap()->setRotation(mEnvRotAngle);
    }
    mpPixelDebug->renderUI(widget);

    // If rendering options that modify the output have changed, set flag to indicate that.
    // In execute() we will pass the flag to other passes for reset of temporal data etc.
    if (dirty)
    {
        mCudaAvgTime = mCudaTime;
        mCudaAccumulatedFrames = 1;
        mOptionsChanged = true;
    }
}

void NeuralMatRendering::loadNetwork(RenderContext* pRenderContext)
{
    mpNBTFFP32 = std::make_shared<NBTF>(mpDevice, "leather08", false);

}

void NeuralMatRendering::setScene(RenderContext* pRenderContext, const ref<Scene>& pScene)
{
    // Clear data for previous scene.
    // After changing scene, the raytracing program should to be recreated.
    mTracer.pProgram = nullptr;
    mTracer.pBindingTable = nullptr;
    mTracer.pVars = nullptr;
    mFrameCount = 0;

    // Set new scene.
    mpScene = pScene;

    if (mpScene)
    {
        if (pScene->hasGeometryType(Scene::GeometryType::Custom))
        {
            logWarning("NeuralMatRendering: This render pass does not support custom primitives.");
        }

        // create envmap sampler
        if (mpScene->useEnvLight())
        {
            if (!mpEnvMapSampler)
            {
                mpEnvMapSampler = std::make_unique<EnvMapSampler>(mpDevice, mpScene->getEnvMap());
            }
        }
        else
        {
            if (mpEnvMapSampler)
            {
                mpEnvMapSampler = nullptr;
            }
        }

        if (mpScene->getEnvMap() != nullptr)
        {
            mEnvRotAngle = mpScene->getEnvMap()->getRotation();
        }

        // Create ray tracing program.
        ProgramDesc desc;
        desc.addShaderModules(mpScene->getShaderModules());
        desc.addShaderLibrary(kShaderFile);
        desc.setMaxPayloadSize(kMaxPayloadSizeBytes);
        desc.setMaxAttributeSize(mpScene->getRaytracingMaxAttributeSize());
        desc.setMaxTraceRecursionDepth(kMaxRecursionDepth);

        mTracer.pBindingTable = RtBindingTable::create(2, 2, mpScene->getGeometryCount());
        auto& sbt = mTracer.pBindingTable;
        sbt->setRayGen(desc.addRayGen("rayGen"));
        sbt->setMiss(0, desc.addMiss("scatterMiss"));
        sbt->setMiss(1, desc.addMiss("shadowMiss"));

        if (mpScene->hasGeometryType(Scene::GeometryType::TriangleMesh))
        {
            sbt->setHitGroup(
                0,
                mpScene->getGeometryIDs(Scene::GeometryType::TriangleMesh),
                desc.addHitGroup("scatterTriangleMeshClosestHit", "scatterTriangleMeshAnyHit")
            );
            sbt->setHitGroup(
                1, mpScene->getGeometryIDs(Scene::GeometryType::TriangleMesh), desc.addHitGroup("", "shadowTriangleMeshAnyHit")
            );
        }

        if (mpScene->hasGeometryType(Scene::GeometryType::DisplacedTriangleMesh))
        {
            sbt->setHitGroup(
                0,
                mpScene->getGeometryIDs(Scene::GeometryType::DisplacedTriangleMesh),
                desc.addHitGroup("scatterDisplacedTriangleMeshClosestHit", "", "displacedTriangleMeshIntersection")
            );
            sbt->setHitGroup(
                1,
                mpScene->getGeometryIDs(Scene::GeometryType::DisplacedTriangleMesh),
                desc.addHitGroup("", "", "displacedTriangleMeshIntersectionShadow")
            );
        }

        mTracer.pProgram = Program::create(mpDevice, desc, mpScene->getSceneDefines());
    }

    mpFence = mpDevice->createFence();
    mpFence->breakStrongReferenceToDevice();

    DefineList defines = mpScene->getSceneDefines();
    mpDisplayPass = ComputePass::create(mpDevice, "RenderPasses/NeuralMatRendering/Display.cs.slang", "csMain", defines);



    // cuda timer
    cudaEventCreate(&mCudaStart);
    cudaEventCreate(&mCudaStop);

    loadNetwork(pRenderContext);
}

void NeuralMatRendering::prepareVars()
{
    FALCOR_ASSERT(mpScene);
    FALCOR_ASSERT(mTracer.pProgram);

    // Configure program.
    mTracer.pProgram->addDefines(mpSampleGenerator->getDefines());
    mTracer.pProgram->setTypeConformances(mpScene->getTypeConformances());

    // Create program variables for the current program.
    // This may trigger shader compilation. If it fails, throw an exception to abort rendering.
    mTracer.pVars = RtProgramVars::create(mpDevice, mTracer.pProgram, mTracer.pBindingTable);

    // Bind utility classes into shared data.
    auto var = mTracer.pVars->getRootVar();
    mpSampleGenerator->bindShaderData(var);
}
