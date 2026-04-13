#pragma once
#include "Core/Macros.h"
#include "Core/API/Buffer.h"
#include "Core/Program/ShaderVar.h"
#include <memory>
#include <random>
#include "MLP.h"
#include "Utils/Texture/Synthesis.h"
namespace Falcor
{
struct FeatureTex{
    int2 texDim;
    ref<Texture> featureTex;
    // for cuda
    // ref<Buffer> featureBuffer;
    std::vector<float> featureData;
};



class FALCOR_API NBTF
{
public:

    NBTF(ref<Device> pDevice, std::string networkPath, bool buildCuda = false);

    void loadFeature(ref<Device> pDevice, std::string featurePath);

    void bindShaderData(const ShaderVar& var) const;

    FeatureTex mHP;
    FeatureTex mDP;
    FeatureTex mUP;
    FeatureTex mTP;
    FeatureTex mTPInv;

    std::unique_ptr<MLP> mpMLP;

    std::unique_ptr<TextureSynthesis> mpTextureSynthesis;
    std::string mNetworkName;

    int mLayerNum;
    int mMaxDim;
    bool mBuildCuda = false;
    bool mHistogram = true;
    // Synthesis parameters
    /// GPU fence for synchronizing readback.
    ref<Fence> mpFence;
};

}
