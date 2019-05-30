// Copyright 2017 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef SERVICES_ML_EXECUTION_IMPL_MAC_MPS_H_
#define SERVICES_ML_EXECUTION_IMPL_MAC_MPS_H_

#import <Metal/MTLBuffer.h>
#import <Metal/MTLCommandBuffer.h>
#include <map>
#include <memory>
#include <vector>

#include "common.h"
#include "compilation_impl_mac.h"

@class MPSImage;
@class MPSTemporaryImage;

namespace ml {

class ExecutionImplMacMPS {
 public:
  ExecutionImplMacMPS(CompilationImplMac*);
  ~ExecutionImplMacMPS();

  void StartCompute();

  bool IsValid() const;
  
  std::vector<float> OutputData();

 private:
  void API_AVAILABLE(macos(10_13))
      SetupMPSImageForOperands(std::vector<MPSImage*>&,
                               std::vector<id<MTLBuffer>>&,
                               const std::vector<uint32_t>&);
  void CreateOutputMTLBuffer();

  API_AVAILABLE(macos(10_13))
  void EncodeCustomKernel(const id<MTLCommandBuffer>& command_buffer,
                          std::map<uint32_t, MPSImage*>& output_mps_images,
                          std::map<uint32_t, MPSImage*>& temporary_mps_images,
                          uint32_t input_index);
  
  void API_AVAILABLE(macos(10_13)) UploadToMPSImage(const MPSImage*,
                                                    const id<MTLBuffer>&,
                                                    const id<MTLCommandBuffer>&,
                                                    const std::vector<float>& data);

  CompilationImplMac* compilation_;

//  std::vector<std::unique_ptr<OperandInfo>> inputs_info_;
//  std::vector<std::unique_ptr<OperandInfo>> outputs_info_;

  API_AVAILABLE(macos(10_13))
  std::vector<MPSImage*> input_mpsimages_;
  API_AVAILABLE(macos(10_13)) std::vector<id<MTLBuffer>> input_mtlbuffers_;
  API_AVAILABLE(macos(10_13)) std::vector<id<MTLBuffer>> output_mtlbuffers_;
  API_AVAILABLE(macos(10_13))
  std::vector<MPSImage*> constant_mpsimages_;
  API_AVAILABLE(macos(10_13)) std::vector<id<MTLBuffer>> constant_mtlbuffers_;
  
  std::vector<float> output_data_;
};

}  // namespace ml

#endif  // SERVICES_ML_EXECUTION_IMPL_MAC_MPS_H_
