//
// Created by terminal on 2020/12/27.
//

#include "NvInfer.h"
#include "l2_norm_plugin.h"

using namespace std;
using namespace nvinfer1;

// plugin specific constants
namespace
{
    static const char *L2_NORM_PLUGIN_VERSION{"1"};
    static const char *L2_NORM_PLUGIN_NAME{"L2_NORM_PLUGIN_Dynamic"};
} // namespace

// Static class fields initialization
PluginFieldCollection L2_Norm_Plugin_Creator::mFC{};
std::vector<PluginField> L2_Norm_Plugin_Creator::mPluginAttributes;

L2_Norm_Plugin_Creator::L2_Norm_Plugin_Creator()
{
    mPluginAttributes.emplace_back(PluginField("op_type", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("eps", nullptr, PluginFieldType::kFLOAT16, 1));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char *L2_Norm_Plugin_Creator::getPluginName() const
{
    return L2_NORM_PLUGIN_NAME;
}

const char *L2_Norm_Plugin_Creator::getPluginVersion() const
{
    return L2_NORM_PLUGIN_VERSION;
}

const PluginFieldCollection *L2_Norm_Plugin_Creator::getFieldNames()
{
    return &mFC;
}

void L2_Norm_Plugin_Creator::setPluginNamespace(const char *pluginNamespace)
{
    mNamespace = pluginNamespace;
}

const char *L2_Norm_Plugin_Creator::getPluginNamespace() const
{
    return mNamespace.c_str();
}

nvinfer1::IPluginV2 *L2_Norm_Plugin_Creator::createPlugin(const char *name, const nvinfer1::PluginFieldCollection *fc)
{
}
nvinfer1::IPluginV2 *L2_Norm_Plugin_Creator::deserializePlugin(const char *name, const void *serialData, size_t serialLength)
{
}