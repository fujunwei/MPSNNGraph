// Copyright 2017 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef SERVICES_ML_COMMON_H_
#define SERVICES_ML_COMMON_H_

#include <stdint.h>
#include <vector>
#include <map>

namespace ml {

uint32_t product(const std::vector<uint32_t>& dims);

struct Operand {
  Operand();
  ~Operand();
  Operand(const Operand&);
  uint32_t requiredSize() const;
  int32_t type;
  std::vector<uint32_t> dimensions;
  float scale;
  int32_t zeroPoint;
};

//struct OperandInfo {
//  // The vaule of input is set by oneself like setOperandValue, so offset = 0
//  // mojo::ScopedSharedBufferMapping => void*
//  OperandInfo(uint32_t length, void* mapping);
//  ~OperandInfo();
//  uint32_t offset;
//  uint32_t length;
//  void* mapping;
//};

struct ValueInfo {
  ValueInfo();
  ~ValueInfo();
  ValueInfo(const ValueInfo&);
  uint32_t index;
//  uint32_t offset;
//  uint32_t length;
  std::vector<float> data;
};

struct Operation {
  Operation();
  ~Operation();
  Operation(const Operation&);
  int32_t type;
  std::vector<uint32_t> inputs;
  std::vector<uint32_t> outputs;
};

int32_t getScalarInt32(const ValueInfo&, int8_t*);
int32_t getScalarInt32(const std::map<uint32_t, ValueInfo>& values,
                       uint32_t key,
                       int8_t* memory);
float getScalarFloat(const ValueInfo&, int8_t*);
float getScalarFloat(const std::map<uint32_t, ValueInfo>& values,
                     uint32_t key,
                     int8_t*);

}

#endif  // SERVICES_ML_COMMON_H_
