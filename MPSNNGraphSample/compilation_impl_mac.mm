// Copyright 2017 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "compilation_impl_mac.h"

#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include "compilation_impl_mac_mps.h"
#include "execution_impl_mac_mps.h"
#include "mps_protocols_impl.h"
#include "mpscnn_context.h"
#include "constants.h"

namespace ml {

CompilationImplMac::CompilationImplMac(ModelImplMac* model)
    : compilation_factory_(this) {
  operands_.reserve(model->operands_.size());
  for (uint32_t i = 0; i < model->operands_.size(); ++i) {
    OperandMac operand(model->operands_[i]);
    operands_.push_back(operand);
  }
  operations_.reserve(model->operations_.size());
  for (uint32_t i = 0; i < model->operations_.size(); ++i) {
    OperationMac operation(model->operations_[i]);
    operations_.push_back(operation);
  }
  values_ = model->values_;
  inputs_ = model->inputs_;
  outputs_ = model->outputs_;
  is_bnns_ = true;
}

CompilationImplMac::~CompilationImplMac() {
}

void CompilationImplMac::Finish(int32_t preference) {
  if (@available(macOS 10.13, *)) {
    CompileModelWithMPS();
  }
}

//void CompilationImplMac::CreateExecution() {
//  auto init_params = ExecutionInitParams::New();
//
//  uint32_t input_memory_size = 0;
//  init_params->inputs.reserve(inputs_.size());
//  for (size_t i = 0; i < inputs_.size(); ++i) {
//    OperandMac& operand = operands_[inputs_[i]];
//    input_memory_size += operand.requiredSize();
//    init_params->inputs.push_back(
//        OperandInfo::New(inputs_[i], operand.type, operand.dimensions));
//  }
//  std::cout << "Required input memory size: " << input_memory_size;
//
//  uint32_t output_memory_size = 0;
//  init_params->outputs.reserve(outputs_.size());
//  for (size_t i = 0; i < outputs_.size(); ++i) {
//    OperandMac& operand = operands_[outputs_[i]];
//    output_memory_size += operand.requiredSize();
//    init_params->outputs.push_back(
//        OperandInfo::New(outputs_[i], operand.type, operand.dimensions));
//  }
//  std::cout << "Required output memory size: " << output_memory_size;
//
//  mojo::ScopedSharedBufferHandle memory_handle =
//      mojo::SharedBufferHandle::Create(input_memory_size + output_memory_size);
//
//  init_params->memory =
//      memory_handle->Clone(mojo::SharedBufferHandle::AccessMode::READ_WRITE);
//
//  ExecutionPtrInfo ptr_info;
//  auto impl = std::make_unique<ExecutionImplMacMPS>(
//      compilation_factory_.GetWeakPtr(), std::move(memory_handle));
//  if (!impl->IsValid()) {
//    return;
//  }
//  mojo::MakeStrongBinding(std::move(impl), mojo::MakeRequest(&ptr_info));
//  
//  init_params->execution = std::move(ptr_info);
//
//  std::move(callback).Run(NOT_ERROR, std::move(init_params));
//}

API_AVAILABLE(macosx(10.13))
void CompilationImplMac::CompileModelWithMPS() {
  if (!GetMPSCNNContext().IsValid()) {
    return;
  }

  // Reset intermediate variable.
  graphs_.clear();
  mps_image_nodes_.clear();

  // Create a placeholder for inputs image.
  for (auto index : inputs_) {
    mps_image_nodes_[index] = [[MPSNNImageNode alloc] initWithHandle:nullptr];
  }

  bool success = true, new_graph = false, custom_kernel = false;
  std::vector<uint32_t> graph_outputs, current_graph_inputs;
  for (size_t i = 0; i < operations_.size(); ++i) {
    OperationMac& operation = operations_[i];
    uint32_t type = operation.type;
    std::vector<uint32_t>& inputs = operation.inputs;
    std::vector<uint32_t>& outputs = operation.outputs;
    // Adjust the read count
    for (size_t j = 0; j < inputs.size(); ++j) {
      OperandMac& operand = operands_[inputs[j]];
      operand.read_count += 1;
    }
      
    // `current_graph_inputs` is use to export image node for inputs of Add
    // operation isn't in current graph.
    current_graph_inputs.push_back(inputs[0]);
    if (new_graph) {
      MPSNNImageNode* export_image_node = mps_image_nodes_[inputs[0]];
      TemporaryImageHandle* input_handle = [[TemporaryImageHandle alloc]
          initWithLabel:[NSString stringWithFormat:@"%d", inputs[0]]];
        // The export node is null if that is custom kernel.
        if (export_image_node) {
            export_image_node.exportFromGraph = true;
            export_image_node.handle = input_handle;
        }
      // Create a placeholder for input image, but mps_image_nodes_[inputs[0]]
      // doesn't need reuse in new graph that does not need to reset.
      mps_image_nodes_[inputs[0]] =
          [[MPSNNImageNode alloc] initWithHandle:input_handle];

      new_graph = false;
      current_graph_inputs.clear();
    }
    current_graph_inputs.push_back(outputs[0]);

    assert(outputs.size() == 1);
    if (type == CONV_2D ||
        type == DEPTHWISE_CONV_2D ||
        type == ATROUS_CONV_2D ||
        type == ATROUS_DEPTHWISE_CONV_2D) {
      success = CompileConv2DOrDepthwiseConv2D(mps_image_nodes_, operation,
                                               values_, memory_, operands_);
    } else if (type == AVERAGE_POOL_2D || type == MAX_POOL_2D) {
      success = CompileAverageOrMaxPool2D(mps_image_nodes_, operation, values_,
                                          memory_, operands_);
    } else if (type == SOFTMAX) {
      success = CompileSoftmax(mps_image_nodes_, operation, values_, memory_);
    } else if (type == RESHAPE) {
      success = CompileReshape(operations_, operation);
    } else if (type == CONCATENATION) {
      success = CompileConcatenation(mps_image_nodes_, operations_, operation,
                                     values_, memory_, operands_, current_graph_inputs);
    } else if (type == ADD || type == MUL) {
      success = CompileArithmetic(mps_image_nodes_, operation, operands_,
                                  constants_, values_, memory_);
    } else if (type == FULLY_CONNECTED) {
      success = CompileFullyConnected(mps_image_nodes_, operation, operands_,
                                      values_, memory_);
    } else if (type == RESIZE_BILINEAR) {
      success = CompileBilinearScale(mps_image_nodes_, operation, custom_kernel,
                                     operands_, values_, memory_);
    } else {
      success = false;
    }

    if (!success)
      break;

    if (custom_kernel) {
      // It's first operation if inputs[0] == inputs_[0] that doesn't need to
      // splite new graph.
      if (inputs[0] != inputs_[0]) {
        MPSNNImageNode* export_image_node = mps_image_nodes_[inputs[0]];
        // And the outputImageNode of first need to be export as temporary
        // image.
        export_image_node.exportFromGraph = true;
        export_image_node.handle = [[TemporaryImageHandle alloc]
                                    initWithLabel:[NSString stringWithFormat:@"%d", inputs[0]]];
        // The custom kernel splite graphs into 2 graphs (first, second).
        // The first graph outputImageNode is the inputs[0] of current
        // operation.
        graph_outputs.push_back(inputs[0]);
      }
      
      custom_operations_[operation.inputs[0]] = operation;
      custom_kernel = false;
      new_graph = true;
    } else {
      for (size_t i = 0; i < outputs_.size(); i++) {
        if (outputs[0] == outputs_[i]) {
          new_graph = true;
          // The order of graph is not the same as outputs_.
          graph_outputs.push_back(outputs[0]);
          // Set index of output image.
          mps_image_nodes_[outputs[0]].handle = [[TemporaryImageHandle alloc]
                                                 initWithLabel:[NSString stringWithFormat:@"%d", outputs[0]]];
        }
      }
    }
  }

  if (success) {
    // The output image need to return result with MPSImage.
    for (size_t i = 0; i < graph_outputs.size(); i++) {
      // OutputImageAllocator* image_allocator = [[OutputImageAllocator alloc]
      // init]; mps_image_nodes_[outputs[0]].imageAllocator = image_allocator;
      // mps_image_nodes_[outputs[0]].exportFromGraph = true;

      // Multiple outputs api initWithDevice:resultImages:resultsAreNeeded: is
      // only available after 10.14.1+.
      if (@available(macOS 10.13.4, *)) {
        // The graph itself is an MPSNNGraph object and is connected to the
        // output of the very last layer in the network
        graphs_.push_back([[MPSNNGraph alloc]
                 initWithDevice:GetMPSCNNContext().device
                    resultImage:mps_image_nodes_[graph_outputs[i]]
            resultImageIsNeeded:true]);
      }
      // DLOG(ERROR) << base::SysNSStringToUTF8([graph_ debugDescription]);
    }
  }
}

}  // namespace ml
