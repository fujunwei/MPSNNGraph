// Copyright 2017 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "test_cases.h"

#include <iostream>
#include <vector>

#include "constants.h"
#include "model_impl_mac.h"
#include "compilation_impl_mac.h"
#include "execution_impl_mac_mps.h"

namespace ml {

void ConvFloat() {
  std::cout << "\nBegin ConvFloat test cases.\n\n";
  std::vector<float> expected_data = {0.875, 0.875, 0.875, 0.875};
  auto model = std::make_unique<ModelImplMac>();
  
  std::vector<uint32_t> dimensions0 = {1, 3, 3, 1};
  std::vector<uint32_t> dimensions1 = {1, 2, 2, 1};
  std::vector<uint32_t> dimensions2 = {1};
  
  uint32_t operandIndex = 0;
  uint32_t op1 = operandIndex++;
  model->AddOperand(TENSOR_FLOAT32, dimensions0, 0, 0);
  uint32_t op2 = operandIndex++;
  model->AddOperand(TENSOR_FLOAT32, dimensions1, 0, 0);
  uint32_t op3 = operandIndex++;
  model->AddOperand(TENSOR_FLOAT32, dimensions2, 0, 0);
  uint32_t pad0 = operandIndex++;
  model->AddOperand(INT32, {}, 0, 0);
  uint32_t act = operandIndex++;
  model->AddOperand(INT32, {}, 0, 0);
  uint32_t stride = operandIndex++;
  model->AddOperand(INT32, {}, 0, 0);
  uint32_t op4 = operandIndex++;
  model->AddOperand(TENSOR_FLOAT32, dimensions1, 0, 0);
  
  model->SetOperandValue(op1, {1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0});
  model->SetOperandValue(op2, {0.25, 0.25, 0.25, 0.25});
  model->SetOperandValue(op3, {0});
  model->SetOperandValue(pad0, {0});
  model->SetOperandValue(act, {0});
  model->SetOperandValue(stride, {1});
  model->AddOperation(CONV_2D, {op1, op2, op3, pad0, pad0, pad0, pad0, stride, stride, act}, {op4});
  
  model->IdentifyInputsAndOutputs({op1}, {op4});
  
  auto compilation = std::make_unique<CompilationImplMac>(model.get());
  compilation->Finish(PREFER_SUSTAINED_SPEED);
  
  auto execution = std::make_unique<ExecutionImplMacMPS>(compilation.get());
  execution->StartCompute();
  
  std::cout << "\n=============================";
  if (execution->OutputData() == expected_data) {
    std::cout << "\nConvFloat test case passed.\n";
  } else {
    std::cout << "\nConvFloat test case doesn't pass.\n";
  }
  std::cout << "=============================\n";
}

}  // namespace ml
