// Copyright 2017 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "common.h"

#include "constants.h"

namespace ml {

uint32_t product(const std::vector<uint32_t>& dims) {
  uint32_t prod = 1;
  for (size_t i = 0; i < dims.size(); ++i) prod *= dims[i];
  return prod;
}

uint32_t GetRequiredSize(int32_t type, const std::vector<uint32_t>& dimensions) {
  if (type == FLOAT32) {
    return sizeof(float);
  } else if (type == INT32) {
    return sizeof(int32_t);
  } else if (type == UINT32) {
    return sizeof(uint32_t);
  } else if (type == TENSOR_FLOAT32) {
    return product(dimensions) * sizeof(float);
  } else if (type == TENSOR_INT32) {
    return product(dimensions) * sizeof(int32_t);
  } else if (type == TENSOR_QUANT8_ASYMM) {
    return product(dimensions) * sizeof(int8_t);
  }
  return 0;
}

Operand::Operand() = default;
Operand::~Operand() = default;
Operand::Operand(const Operand&) = default;
uint32_t Operand::requiredSize() const {
  return GetRequiredSize(type, dimensions);
}

Operation::Operation() = default;
Operation::~Operation() = default;
Operation::Operation(const Operation&) = default;
//
//OperandInfo::OperandInfo(uint32_t length, void* mapping) :
//      length(length), mapping(std::move(mapping)) {}
//
//OperandInfo::~OperandInfo() {}

ValueInfo::ValueInfo() = default;
ValueInfo::~ValueInfo() = default;
ValueInfo::ValueInfo(const ValueInfo&) = default;

int32_t getScalarInt32(const ValueInfo& info, int8_t* memory) {
  return info.data[0];
}

// There are no viable overloaded operator[] for type
// 'const std::map<uint32_t, ValueInfo>', so uses 'find' instead of it.
int32_t getScalarInt32(const std::map<uint32_t, ValueInfo>& values,
                       uint32_t key,
                       int8_t* memory) {
  auto iter = values.find(key);
  if (iter == values.end()) {
    assert(0);
    return -1;
  }

  return getScalarInt32(iter->second, memory);
}

float getScalarFloat(const ValueInfo& info, int8_t* memory) {
  return info.data[0];
}

float getScalarFloat(const std::map<uint32_t, ValueInfo>& values,
                     uint32_t key,
                     int8_t* memory) {
  auto iter = values.find(key);
  if (iter == values.end()) {
    assert(0);
    return -1.0;
  }

  return getScalarFloat(iter->second, memory);
}

}
