// Copyright 2017 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef SERVICES_ML_COMPILATION_IMPL_MAC_MPS_H_
#define SERVICES_ML_COMPILATION_IMPL_MAC_MPS_H_

#include <map>
#include <memory>
#include <vector>
#include <AvailabilityMacros.h>
#import <os/availability.h>

#include "common.h"
#include "ml_utils_mac.h"

class CompilationImplMac;
@class MPSNNImageNode;
@class MPSImageDescriptor;

namespace ml {


bool CompileConv2DOrDepthwiseConv2D(
    std::map<uint32_t, MPSNNImageNode*>& image_nodes,
    const OperationMac&,
    const std::map<uint32_t, ValueInfo>& values,
    std::unique_ptr<int8_t[]>& memory,
    const std::vector<OperandMac>& operands);


bool CompileAverageOrMaxPool2D(std::map<uint32_t, MPSNNImageNode*>& image_nodes,
                               const OperationMac& operation,
                               const std::map<uint32_t, ValueInfo>& values,
                               const std::unique_ptr<int8_t[]>& memory,
                               const std::vector<OperandMac>& operands);


bool CompileSoftmax(std::map<uint32_t, MPSNNImageNode*>& image_nodes,
                    const OperationMac& operation,
                    const std::map<uint32_t, ValueInfo>& values,
                    const std::unique_ptr<int8_t[]>& memory);

bool CompileReshape(std::vector<OperationMac>& operations,
                    const OperationMac& reshape);


bool CompileConcatenation(std::map<uint32_t, MPSNNImageNode*>& image_nodes,
                          std::vector<OperationMac>& operations,
                          const OperationMac& concat,
                          const std::map<uint32_t, ValueInfo>& values,
                          const std::unique_ptr<int8_t[]>& memory,
                          const std::vector<OperandMac>& operands,
                          const std::vector<uint32_t>& current_graph_inputs);


bool CompileArithmetic(std::map<uint32_t, MPSNNImageNode*>& image_nodes,
                       const OperationMac& operation,
                       const std::vector<OperandMac>& operands,
                       std::vector<uint32_t>& constants,
                       const std::map<uint32_t, ValueInfo>& values,
                       const std::unique_ptr<int8_t[]>& memory);


bool CompileFullyConnected(std::map<uint32_t, MPSNNImageNode*>& image_nodes,
                           OperationMac&,
                           std::vector<OperandMac>& operands,
                           const std::map<uint32_t, ValueInfo>& values,
                           const std::unique_ptr<int8_t[]>& memory);


bool CompileBilinearScale(std::map<uint32_t, MPSNNImageNode*>& image_nodes,
                          OperationMac&,
                          bool& custom_kernel,
                          const std::vector<OperandMac>& operands,
                          const std::map<uint32_t, ValueInfo>& values,
                          const std::unique_ptr<int8_t[]>& memory);

}  // namespace ml

#endif  // SERVICES_ML_COMPILATION_IMPL_MAC_MPS_H_
