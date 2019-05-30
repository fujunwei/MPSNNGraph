// Copyright 2017 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "resize_bilinear_test.h"

#include <iostream>
#include <vector>
#include <math.h>

#include "constants.h"
#include "model_impl_mac.h"
#include "compilation_impl_mac.h"
#include "execution_impl_mac_mps.h"

namespace ml {

void ResizeBilinear65_65To513_513() {
  auto model = std::make_unique<ModelImplMac>();
  uint32_t operandIndex = 0;
  
  NSString* input_data_path = [[NSBundle mainBundle] pathForResource: [[NSString alloc] initWithUTF8String: "input_data_65_65_1"] ofType: @"txt"];
  std::vector<float> input_data = LoadData(input_data_path);
  NSString* expected_data_path = [[NSBundle mainBundle] pathForResource: [[NSString alloc] initWithUTF8String: "output_data_513_513_1"] ofType: @"txt"];
  std::vector<float> expected_data = LoadData(expected_data_path);;
  
  std::vector<uint32_t> dimensions0 = {1, 65, 65, 1};
  std::vector<uint32_t> dimensions1 = {1, 513, 513, 1};
  
  uint32_t input_index = operandIndex++;
  model->AddOperand(TENSOR_FLOAT32, dimensions0, 0, 0);
  uint32_t output_index = operandIndex++;
  model->AddOperand(TENSOR_FLOAT32, dimensions1, 0, 0);
  uint32_t width_index = operandIndex++;
  model->AddOperand(INT32, {}, 0, 0);
  uint32_t height_index = operandIndex++;
  model->AddOperand(INT32, {}, 0, 0);
  
  model->SetOperandValue(input_index, input_data);
  model->SetOperandValue(width_index, {513});
  model->SetOperandValue(height_index, {513});
  model->AddOperation(RESIZE_BILINEAR, {input_index, width_index, height_index}, {output_index});
  
  model->IdentifyInputsAndOutputs({input_index}, {output_index});
  
  auto compilation = std::make_unique<CompilationImplMac>(model.get());
  compilation->Finish(PREFER_SUSTAINED_SPEED);
  
  auto execution = std::make_unique<ExecutionImplMacMPS>(compilation.get());
  execution->StartCompute();
  
  int length = 513 * 513;
  std::vector<float> output_data = execution->OutputData();
  std::cout << "\n=============================\n";
  float sum = 0;
  for (int i = 0; i < length; i++) {
    sum += pow(output_data[i] - expected_data[i], 2);
  }
  std::cout << "ResizeBilinear65_65To513_513 test case = " << sum / length << "\n";
  
  for (int i = 0; i < length; ++i) {
    if(output_data[i] != expected_data[i]) {
      std::cout << "index: " << i << " output_data: " << output_data[i] << " expected_data: " << expected_data[i];
      break;
    }
  }
  std::cout << "\n=============================\n";
}
  
void ResizeBilinear65_65_21To513_513_21() {
  auto model = std::make_unique<ModelImplMac>();
  uint32_t operandIndex = 0;
  
  NSString* input_data_path = [[NSBundle mainBundle] pathForResource: [[NSString alloc] initWithUTF8String: "input_data_65_65_21"] ofType: @"txt"];
  std::vector<float> input_data = LoadData(input_data_path);
  NSString* expected_data_path = [[NSBundle mainBundle] pathForResource: [[NSString alloc] initWithUTF8String: "output_data_513_513_21"] ofType: @"txt"];
  std::vector<float> expected_data = LoadData(expected_data_path);;
  
  std::vector<uint32_t> dimensions0 = {1, 65, 65, 21};
  std::vector<uint32_t> dimensions1 = {1, 513, 513, 21};
  
  uint32_t input_index = operandIndex++;
  model->AddOperand(TENSOR_FLOAT32, dimensions0, 0, 0);
  uint32_t output_index = operandIndex++;
  model->AddOperand(TENSOR_FLOAT32, dimensions1, 0, 0);
  uint32_t width_index = operandIndex++;
  model->AddOperand(INT32, {}, 0, 0);
  uint32_t height_index = operandIndex++;
  model->AddOperand(INT32, {}, 0, 0);
  
  model->SetOperandValue(input_index, input_data);
  model->SetOperandValue(width_index, {513});
  model->SetOperandValue(height_index, {513});
  model->AddOperation(RESIZE_BILINEAR, {input_index, width_index, height_index}, {output_index});
  
  model->IdentifyInputsAndOutputs({input_index}, {output_index});
  
  auto compilation = std::make_unique<CompilationImplMac>(model.get());
  compilation->Finish(PREFER_SUSTAINED_SPEED);
  
  auto execution = std::make_unique<ExecutionImplMacMPS>(compilation.get());
  execution->StartCompute();
  
  int length = 513 * 513 * 21;
  std::vector<float> output_data = execution->OutputData();
  std::cout << "\n=============================\n";
  float sum = 0;
  for (int i = 0; i < length; i++) {
    sum += pow(output_data[i] - expected_data[i], 2);
  }
  std::cout << "ResizeBilinear65_65_21To513_513_21 test case = " << sum / length << "\n";
  
  for (int i = 0; i < length; ++i) {
    if(output_data[i] != expected_data[i]) {
      std::cout << "index: " << i << " output_data: " << output_data[i] << " expected_data: " << expected_data[i];
      break;
    }
  }
  std::cout << "\n=============================\n";
}

}  // namespace ml
