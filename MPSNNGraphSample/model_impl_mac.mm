// Copyright 2017 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "model_impl_mac.h"

#include <iostream>

#include "compilation_impl_mac.h"
#include "constants.h"

namespace ml {

ModelImplMac::ModelImplMac() = default;
ModelImplMac::~ModelImplMac() = default;

int32_t ModelImplMac::AddOperand(int32_t type, const std::vector<uint32_t>& dimensions, float scale, int32_t zeroPoint) {
  Operand operand;
  operand.type = type;
  operand.dimensions = dimensions;
  operand.scale = scale;
  operand.zeroPoint = zeroPoint;
  operands_.push_back(operand);
  return NOT_ERROR;
}

int32_t ModelImplMac::SetOperandValue(uint32_t index, const std::vector<float>& data) {
  ValueInfo value;
  value.index = index;
  value.data = data;
  values_[index] = value;
  
  return NOT_ERROR;
}

int32_t ModelImplMac::AddOperation(int32_t type, const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs) {
  Operation operation;
  operation.type = type;
  operation.inputs = inputs;
  operation.outputs = outputs;
  operations_.push_back(operation);
  return NOT_ERROR;
}

int32_t ModelImplMac::IdentifyInputsAndOutputs(const std::vector<uint32_t>& inputs, const std::vector<uint32_t>& outputs) {
  inputs_ = inputs;
  outputs_ = outputs;
  return NOT_ERROR;
}

//void ModelImplMac::Finish(mojom::ModelInfoPtr model_info,
//                          FinishCallback callback) {
//  std::cout << "ModelImplMac::Finish";
//  std::cout << "operands(" << model_info->operands.size() << ")";
//  for (size_t i = 0; i < model_info->operands.size(); ++i ) {
//    std::cout << "  operand[" << i << "]";
//    const mojom::OperandPtr& operand = model_info->operands[i];
//    AddOperand(operand->type, operand->dimensions, operand->scale, operand->zeroPoint);
//  }
//  std::cout << "operations(" << model_info->operations.size() << ")";
//  for (size_t i = 0; i < model_info->operations.size(); ++i ) {
//    std::cout << "  operation[" << i << "]";
//    const mojom::OperationPtr& operation = model_info->operations[i];
//    AddOperation(operation->type, operation->inputs, operation->outputs);
//  }
//  std::cout << "values(" << model_info->values.size() << ")";
//  memory_size_ = model_info->memory_size;
//  auto mapping = model_info->memory->Map(memory_size_);
//  const int8_t* base = static_cast<const int8_t*>(mapping.get());
//  memory_.reset(new int8_t[memory_size_]);
//  memcpy(memory_.get(), base, memory_size_);
//  for (auto itr = model_info->values.begin(); itr != model_info->values.end();
//       ++itr) {
//    const mojom::OperandValueInfoPtr& value_info = itr->second;
//    int32_t result = SetOperandValue(value_info->index,
//                                     static_cast<const void*>(memory_.get() + value_info->offset),
//                                     value_info->length);
//    if (result != NOT_ERROR) {
//      std::move(callback).Run(result);
//      return;
//    }
//    ValueInfo value;
//    value.index = value_info->index;
//    value.offset = value_info->offset;
//    value.length = value_info->length;
//    values_[value_info->index] = value;
//  }
//  std::cout << "inputs(" << model_info->inputs.size() << ")";
//  std::cout << "outputs(" << model_info->outputs.size() << ")";
//  IdentifyInputsAndOutputs(model_info->inputs, model_info->outputs);
//
//  std::move(callback).Run(NOT_ERROR);
//}
//
//CompilationImplMac* ModelImplMac::CreateCompilation() {
//  std::cout << "ModelImplMac::CreateCompilation";
//
//}

}  // namespace ml
