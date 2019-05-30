// Copyright 2017 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "depthwise_conv_test.h"

#include <iostream>
#include <vector>
#include <math.h>

#include "constants.h"
#include "model_impl_mac.h"
#include "compilation_impl_mac.h"
#include "execution_impl_mac_mps.h"

namespace ml {

void DepthwiseConv2dFloatLarge() {
  std::cout << "\nBegin DepthwiseConv2dFloatLarge test cases.\n\n";
  auto model = std::make_unique<ModelImplMac>();
  uint32_t operandIndex = 0;
  
  std::vector<float> op1_value = {10, 21, 10, 22, 10, 23, 10, 24};
  std::vector<float> expected_data = {110, 246};
  
  std::vector<uint32_t> dimensions0 = {1, 2, 2, 2};
  std::vector<uint32_t> dimensions1 = {2};
  std::vector<uint32_t> dimensions3 = {1, 1, 1, 2};
  
  uint32_t op1 = operandIndex++;
  model->AddOperand(TENSOR_FLOAT32, dimensions0, 0, 0);
  uint32_t op2 = operandIndex++;
  model->AddOperand(TENSOR_FLOAT32, dimensions0, 0, 0);
  uint32_t op3 = operandIndex++;
  model->AddOperand(TENSOR_FLOAT32, dimensions1, 0, 0);
  uint32_t pad0 = operandIndex++;
  model->AddOperand(INT32, {}, 0, 0);
  uint32_t act = operandIndex++;
  model->AddOperand(INT32, {}, 0, 0);
  uint32_t stride = operandIndex++;
  model->AddOperand(INT32, {}, 0, 0);
  uint32_t channelMultiplier = operandIndex++;
  model->AddOperand(INT32, {}, 0, 0);
  uint32_t op4 = operandIndex++;
  model->AddOperand(TENSOR_FLOAT32, dimensions3, 0, 0);
  
  model->SetOperandValue(op1, op1_value);
  model->SetOperandValue(op2, {0.25, 0, 0.25, 1, 0.25, 0, 0.25, 1});
  model->SetOperandValue(op3, {100, 200});
  model->SetOperandValue(pad0, {0});
  model->SetOperandValue(act, {0});
  model->SetOperandValue(stride, {1});
  model->SetOperandValue(channelMultiplier, {1});
  model->AddOperation(DEPTHWISE_CONV_2D, {op1, op2, op3, pad0, pad0, pad0, pad0, stride, stride, channelMultiplier, act}, {op4});
  
  model->IdentifyInputsAndOutputs({op1}, {op4});
  
  auto compilation = std::make_unique<CompilationImplMac>(model.get());
  compilation->Finish(PREFER_SUSTAINED_SPEED);
  
  auto execution = std::make_unique<ExecutionImplMacMPS>(compilation.get());
  execution->StartCompute();
  
  std::cout << "\n=============================";
  if (execution->OutputData() == expected_data) {
    std::cout << "\nDepthwiseConv2dFloatLarge test case passed.\n";
  } else {
    std::cout << "\nDepthwiseConv2dFloatLarge test case doesn't pass.\n";
  }
  std::cout << "=============================\n";
}
  
void Depthwise28_28Conv5_5() {
  auto model = std::make_unique<ModelImplMac>();
  uint32_t operandIndex = 0;
  
  NSString* input_data_path = [[NSBundle mainBundle] pathForResource: [[NSString alloc] initWithUTF8String: "input_data_28_28_1"] ofType: @"txt"];
  std::vector<float> input_data = LoadData(input_data_path);
  NSString* weights_path = [[NSBundle mainBundle] pathForResource: [[NSString alloc] initWithUTF8String: "weights_5_5_1"] ofType: @"txt"];
  std::vector<float> weights_data = LoadData(weights_path);
  NSString* expected_data_path = [[NSBundle mainBundle] pathForResource: [[NSString alloc] initWithUTF8String: "depthwise_28_28_conv_5_5_output"] ofType: @"txt"];
  std::vector<float> expected_data = LoadData(expected_data_path);;
  
  std::vector<uint32_t> dimensions0 = {1, 28, 28, 1};
  std::vector<uint32_t> dimensions1 = {1, 5, 5, 1};
  std::vector<uint32_t> dimensions2 = {1};
  
  uint32_t input_index = operandIndex++;
  model->AddOperand(TENSOR_FLOAT32, dimensions0, 0, 0);
  uint32_t weights_index = operandIndex++;
  model->AddOperand(TENSOR_FLOAT32, dimensions1, 0, 0);
  uint32_t bias_index = operandIndex++;
  model->AddOperand(TENSOR_FLOAT32, dimensions2, 0, 0);
  uint32_t padding_code = operandIndex++;
  model->AddOperand(INT32, {}, 0, 0);
  uint32_t stride_width = operandIndex++;
  model->AddOperand(INT32, {}, 0, 0);
  uint32_t stride_height = operandIndex++;
  model->AddOperand(INT32, {}, 0, 0);
  uint32_t channelMultiplier = operandIndex++;
  model->AddOperand(INT32, {}, 0, 0);
  uint32_t fuse_code = operandIndex++;
  model->AddOperand(INT32, {}, 0, 0);
  uint32_t output_index = operandIndex++;
  model->AddOperand(TENSOR_FLOAT32, dimensions0, 0, 0);
  
  model->SetOperandValue(input_index, input_data);
  model->SetOperandValue(weights_index, weights_data);
  model->SetOperandValue(bias_index, {1});
  model->SetOperandValue(padding_code, {1});
  model->SetOperandValue(stride_width, {1});
  model->SetOperandValue(stride_height, {1});
  model->SetOperandValue(channelMultiplier, {1});
  model->SetOperandValue(fuse_code, {0});
  model->AddOperation(DEPTHWISE_CONV_2D, {input_index, weights_index, bias_index, padding_code, stride_width, stride_height, channelMultiplier, fuse_code}, {output_index});
  
  model->IdentifyInputsAndOutputs({input_index}, {output_index});
  
  auto compilation = std::make_unique<CompilationImplMac>(model.get());
  compilation->Finish(PREFER_SUSTAINED_SPEED);
  
  auto execution = std::make_unique<ExecutionImplMacMPS>(compilation.get());
  execution->StartCompute();
  
  int length = 28 * 28;
  std::vector<float> output_data = execution->OutputData();
  std::cout << "\n=============================\n";
  float sum = 0;
  for (int i = 0; i < length; i++) {
    sum += pow(output_data[i] - expected_data[i], 2);
  }
  std::cout << "Depthwise28_28Conv5_5 test case = " << sum / length << "\n";
  
  for (int i = 0; i < length; ++i) {
    if(output_data[i] != expected_data[i]) {
      std::cout << "index: " << i << " output_data: " << output_data[i] << " expected_data: " << expected_data[i];
      break;
    }
  }
  std::cout << "\n=============================\n";
}

void Depthwise28_28_528Conv5_5_528() {
  auto model = std::make_unique<ModelImplMac>();
  uint32_t operandIndex = 0;
  
  NSString* input_data_path = [[NSBundle mainBundle] pathForResource: [[NSString alloc] initWithUTF8String: "input_data_28_28_528"] ofType: @"txt"];
  std::vector<float> input_data = LoadData(input_data_path);
  NSString* weights_path = [[NSBundle mainBundle] pathForResource: [[NSString alloc] initWithUTF8String: "weights_5_5_528"] ofType: @"txt"];
  std::vector<float> weights_data = LoadData(weights_path);
  NSString* bias_path = [[NSBundle mainBundle] pathForResource: [[NSString alloc] initWithUTF8String: "bias_528"] ofType: @"txt"];
  std::vector<float> bias_data = LoadData(bias_path);;
  NSString* expected_data_path = [[NSBundle mainBundle] pathForResource: [[NSString alloc] initWithUTF8String: "depthwise_28_28_528_cov_5_5_528_output"] ofType: @"txt"];
  std::vector<float> expected_data = LoadData(expected_data_path);;
  
  std::vector<uint32_t> dimensions0 = {1, 28, 28, 528};
  std::vector<uint32_t> dimensions1 = {1, 5, 5, 528};
  std::vector<uint32_t> dimensions2 = {528};
  
  uint32_t input_index = operandIndex++;
  model->AddOperand(TENSOR_FLOAT32, dimensions0, 0, 0);
  uint32_t weights_index = operandIndex++;
  model->AddOperand(TENSOR_FLOAT32, dimensions1, 0, 0);
  uint32_t bias_index = operandIndex++;
  model->AddOperand(TENSOR_FLOAT32, dimensions2, 0, 0);
  uint32_t padding_code = operandIndex++;
  model->AddOperand(INT32, {}, 0, 0);
  uint32_t stride_width = operandIndex++;
  model->AddOperand(INT32, {}, 0, 0);
  uint32_t stride_height = operandIndex++;
  model->AddOperand(INT32, {}, 0, 0);
  uint32_t channelMultiplier = operandIndex++;
  model->AddOperand(INT32, {}, 0, 0);
  uint32_t fuse_code = operandIndex++;
  model->AddOperand(INT32, {}, 0, 0);
  uint32_t output_index = operandIndex++;
  model->AddOperand(TENSOR_FLOAT32, dimensions0, 0, 0);
  
  model->SetOperandValue(input_index, input_data);
  model->SetOperandValue(weights_index, weights_data);
  model->SetOperandValue(bias_index, bias_data);
  model->SetOperandValue(padding_code, {1});
  model->SetOperandValue(stride_width, {1});
  model->SetOperandValue(stride_height, {1});
  model->SetOperandValue(channelMultiplier, {1});
  model->SetOperandValue(fuse_code, {1});
  model->AddOperation(DEPTHWISE_CONV_2D, {input_index, weights_index, bias_index, padding_code, stride_width, stride_height, channelMultiplier, fuse_code}, {output_index});
  
  model->IdentifyInputsAndOutputs({input_index}, {output_index});
  
  auto compilation = std::make_unique<CompilationImplMac>(model.get());
  compilation->Finish(PREFER_SUSTAINED_SPEED);
  
  auto execution = std::make_unique<ExecutionImplMacMPS>(compilation.get());
  execution->StartCompute();
  
  std::cout << "\n=============================\n";
  float sum = 0;
  uint32_t length = 28 * 28 * 528;
  std::vector<float> output_data = execution->OutputData();
  for (int i = 0; i < length; i++) {
    sum += pow(output_data[i] - expected_data[i], 2);
  }
  std::cout << "Depthwise28_28_528Conv5_5_528 test case = " << sum / length;
  std::cout << "\n=============================\n";
}
}  // namespace ml
