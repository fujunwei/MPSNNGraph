// Copyright 2017 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef SERVICES_ML_COMPILATION_IMPL_MAC_H_
#define SERVICES_ML_COMPILATION_IMPL_MAC_H_

#include <map>
#include <memory>
#include <vector>

#include "ml_utils_mac.h"
#include "model_impl_mac.h"

@class MPSNNGraph;
@class MPSNNImageNode;

namespace ml {

class ExecutionImplMac;

class CompilationImplMac {
 public:
  explicit CompilationImplMac(ModelImplMac*);
  ~CompilationImplMac();

  void Finish(int32_t preference);
//  void CreateExecution();

 private:
  void CompileModelWithMPS();
  friend class ExecutionImplMacBNNS;
  friend class ExecutionImplMacMPS;

  std::vector<OperandMac> operands_;
  std::vector<OperationMac> operations_;
  std::map<uint32_t, ValueInfo> values_;
  std::vector<uint32_t> inputs_;
  std::vector<uint32_t> outputs_;
  std::vector<uint32_t> constants_;
  std::unique_ptr<int8_t[]> memory_;
  uint32_t memory_size_;
  bool is_bnns_;

  // Used for MPSNNGraph
  std::vector<MPSNNGraph*> graphs_;
  std::map<uint32_t, MPSNNImageNode*> mps_image_nodes_;
  // The first Key is index of input image.
  std::map<uint32_t, OperationMac> custom_operations_;

  CompilationImplMac* compilation_factory_;
};

}  // namespace ml

#endif  // SERVICES_ML_COMPILATION_IMPL_MAC_H_
