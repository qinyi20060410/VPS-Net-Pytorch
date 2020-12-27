//
// Created by terminal on 2020/12/27.
//

#ifndef DEPLOY_L2_NORM_PLUGIN_H
#define DEPLOY_L2_NORM_PLUGIN_H

#include "cuda.h"
#include "NvInferPlugin.h"
#include <string>
#include <vector>

class L2_Norm_Plugin : public nvinfer1::IPluginV2DynamicExt
{
public:
  L2_Norm_Plugin(int op_type, float eps);
  L2_Norm_Plugin(int op_type, float eps, int C, int H, int W);
  L2_Norm_Plugin(const char *buffer, size_t length);
  ~L2_Norm_Plugin() override = default;
};

class L2_Norm_Plugin_Creator : public nvinfer1::IPluginCreator
{
public:
  L2_Norm_Plugin_Creator();
  ~L2_Norm_Plugin_Creator() override = default;

  const char *getPluginName() const override;
  const char *getPluginVersion() const override;

  const nvinfer1::PluginFieldCollection *getFieldNames() override;

  void setPluginNamespace(const char *pluginNamespace) override;
  const char *getPluginNamespace() const override;

  nvinfer1::IPluginV2 *createPlugin(const char *name, const nvinfer1::PluginFieldCollection *fc) override;
  nvinfer1::IPluginV2 *deserializePlugin(const char *name, const void *serialData, size_t serialLength) override;

private:
  static nvinfer1::PluginFieldCollection mFC;
  static std::vector<nvinfer1::PluginField> mPluginAttributes;
  std::string mNamespace;
};

#endif // DEPLOY_L2_NORM_PLUGIN_H
