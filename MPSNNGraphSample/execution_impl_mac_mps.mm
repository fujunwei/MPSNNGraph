// Copyright 2017 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "execution_impl_mac_mps.h"

#include <iostream>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include "ml_utils_mac.h"
#include "mps_protocols_impl.h"
#include "mpscnn_context.h"
#include "constants.h"

namespace ml {

namespace {

NSString* API_AVAILABLE(macosx(10.13)) KernelFor(const MPSImage* X,
                                                 NSString* arrayKernel,
                                                 NSString* nonArrayKernel) {
  if (X.featureChannels > 4) {
    return arrayKernel;
  }
  if (X.numberOfImages > 1) {
    return arrayKernel;
  }
  return nonArrayKernel;
}

auto divRoundUp(uint x, uint y) -> uint {
  return (x + y - 1) / y;
}

struct LaunchParams {
  MTLSize threadsPerThreadgroup;
  MTLSize threadgroupsPerGrid;
};

LaunchParams API_AVAILABLE(macosx(10.13))
    SpatialPointwiseKernelLaunchParams(id<MTLComputePipelineState> pipeline,
                                       const MPSImage* im) {
  // const auto maxThreadsPerThreadgroup =
  //[pipeline maxTotalThreadsPerThreadgroup];
  // const auto threadExecutionWidth = [pipeline threadExecutionWidth];
  const auto threadsPerThreadgroup =
      MTLSizeMake(8 /* threadExecutionWidth */,
                  4 /* maxThreadsPerThreadgroup / threadExecutionWidth */, 1);
  const auto threadgroupsPerGrid =
      MTLSizeMake(divRoundUp(im.width, threadsPerThreadgroup.width),
                  divRoundUp(im.height, threadsPerThreadgroup.height),
                  im.numberOfImages * divRoundUp(im.featureChannels, 4));
  return {threadsPerThreadgroup, threadgroupsPerGrid};
};

MPSImageDescriptor* API_AVAILABLE(macosx(10.13))
    CreateMPSImageDescriptor(const OperandMac& operand) {
  int32_t type = operand.type;
  MPSImageDescriptor* mpsimage_desc = nullptr;
  if (type != TENSOR_FLOAT32) {
    std::cout << "type " << type << " is not supported";
    return mpsimage_desc;
  }
  uint32_t n, width, height, channels;
  if (!ml::GetMPSImageInfo(operand, n, width, height, channels)) {
    return mpsimage_desc;
  }
  mpsimage_desc = [MPSImageDescriptor
      imageDescriptorWithChannelFormat:MPSImageFeatureChannelFormatFloat16
                                 width:width
                                height:height
                       featureChannels:channels
                        numberOfImages:n
                                 usage:MTLTextureUsageShaderRead |
                                       MTLTextureUsageShaderWrite];
  return mpsimage_desc;
}

API_AVAILABLE(macosx(10.13))
void SaveTemporaryImages(std::map<uint32_t, MPSImage*>& temporary_images,
                         const NSMutableArray<MPSImage*>* intermediate_images) {
  for (MPSImage* image in intermediate_images) {
    uint32_t input_index = [image.label intValue];
    temporary_images[input_index] = image;
  }
}

API_AVAILABLE(macosx(10.13))
MPSTemporaryImage* CreateMPSTemporaryImage(
                                           const id<MTLCommandBuffer>& command_buffer,
                                           const OperandMac& operand) {
  MPSImageDescriptor* descriptor = CreateMPSImageDescriptor(operand);
  if (!descriptor)
    return nullptr;
  
  MPSTemporaryImage* temp_image =
  [MPSTemporaryImage temporaryImageWithCommandBuffer:command_buffer
                                     imageDescriptor:descriptor];
  temp_image.readCount = operand.read_count;
  
  return temp_image;
}

API_AVAILABLE(macosx(10.13))
MPSImage* CreateMPSOutputImage(const id<MTLCommandBuffer>& command_buffer,
                               const OperandMac& operand) {
  MPSImageDescriptor* descriptor = CreateMPSImageDescriptor(operand);
  if (!descriptor)
    return nullptr;
  
  return [[MPSImage alloc] initWithDevice:GetMPSCNNContext().device
                          imageDescriptor:descriptor];
}
  
}  // namespace

ExecutionImplMacMPS::ExecutionImplMacMPS(
    CompilationImplMac* compilation) {
  compilation_ = compilation;
  // Inputs data with setOperandValue.
//  SetupOperandInfoForOperands(inputs_info_, compilation_->operands_,
//                              compilation_->inputs_, compilation_->values_,
//                              compilation_->memory_);
//  SetupOperandInfoForOperands(outputs_info_, compilation_->operands_,
//                              compilation_->outputs_, compilation_->values_,
//                              compilation_->memory_);

  if (@available(macOS 10.13, *)) {
    SetupMPSImageForOperands(input_mpsimages_, input_mtlbuffers_,
                             compilation_->inputs_);
    SetupMPSImageForOperands(constant_mpsimages_, constant_mtlbuffers_,
                             compilation_->constants_);
    CreateOutputMTLBuffer();
  }
}

ExecutionImplMacMPS::~ExecutionImplMacMPS() = default;

bool ExecutionImplMacMPS::IsValid() const {
  bool valid = true;
  if (compilation_) {
    if (@available(macOS 10.13, *)) {
      valid &= compilation_->inputs_.size() == input_mpsimages_.size() &&
               compilation_->constants_.size() == constant_mpsimages_.size();
    }
  }
  return valid;
}

void API_AVAILABLE(macosx(10.13)) ExecutionImplMacMPS::SetupMPSImageForOperands(
    std::vector<MPSImage*>& mps_image_array,
    std::vector<id<MTLBuffer>>& mtl_buffer_array,
    const std::vector<uint32_t>& operands_index_array) {
  for (size_t i = 0; i < operands_index_array.size(); ++i) {
    const OperandMac& operand =
        compilation_->operands_[operands_index_array[i]];
    if (@available(macOS 10.13, *)) {
      MPSImageDescriptor* descriptor = CreateMPSImageDescriptor(operand);
      if (!descriptor)
        return;
      MPSImage* mps_img([[MPSImage alloc]
           initWithDevice:GetMPSCNNContext().device
          imageDescriptor:descriptor]);
      mps_image_array.push_back(std::move(mps_img));
      mtl_buffer_array.push_back([GetMPSCNNContext().device
          newBufferWithLength:operand.requiredSize()
                      options:MTLResourceOptionCPUCacheModeWriteCombined]);
    }
  }
}

API_AVAILABLE(macosx(10.13))
void ExecutionImplMacMPS::CreateOutputMTLBuffer() {
  for (size_t i = 0; i < compilation_->outputs_.size(); ++i) {
    const OperandMac& operand =
        compilation_->operands_[compilation_->outputs_[i]];
    output_mtlbuffers_.push_back([GetMPSCNNContext().device
        newBufferWithLength:operand.requiredSize()
                    options:MTLResourceOptionCPUCacheModeWriteCombined]);
  }
}
  
API_AVAILABLE(macosx(10.13))
void ExecutionImplMacMPS::EncodeCustomKernel(
                                             const id<MTLCommandBuffer>& command_buffer,
                                             std::map<uint32_t, MPSImage*>& output_mps_images,
                                             std::map<uint32_t, MPSImage*>& temporary_mps_images,
                                             uint32_t input_index) {
  if (compilation_->custom_operations_.find(input_index) !=
      compilation_->custom_operations_.end()) {
    const OperationMac& custom_operation =
    compilation_->custom_operations_[input_index];
    bool last_node = false;
    for (auto index : compilation_->outputs_) {
      if (custom_operation.outputs[0] == index)
        last_node = true;
    }
    MPSImage* dest_image;
    uint32_t output_index = custom_operation.outputs[0];
    const OperandMac& operand = compilation_->operands_[output_index];
    if (last_node) {
      dest_image = CreateMPSOutputImage(command_buffer, operand);
      output_mps_images[output_index] = dest_image;
    } else {
      dest_image = CreateMPSTemporaryImage(command_buffer, operand);
      temporary_mps_images[output_index] = dest_image;
    }
    
    MPSImage* src_image;
    uint32_t input_index = custom_operation.inputs[0];
    const OperandMac& input_operand = compilation_->operands_[input_index];
    // DCHECK(compilation_->inputs_.size() == 1);
    if (input_index == compilation_->inputs_[0]) {
      src_image = input_mpsimages_[input_index];
    } else {
      // The custom kernel split graph into two sub graphs.
      src_image = temporary_mps_images[input_index];
    }
    [custom_operation.custom_cnn_kernel encodeToCommandBuffer:command_buffer
                                                  sourceImage:src_image
                                             destinationImage:dest_image];
  }
}

void ExecutionImplMacMPS::StartCompute() {
  bool success = true;
  if (@available(macOS 10.13, *)) {
    do {
      @autoreleasepool {
        id<MTLCommandBuffer> command_buffer =
            [GetMPSCNNContext().command_queue commandBuffer];

        NSMutableArray<MPSImage*>* image_array =
            [NSMutableArray arrayWithCapacity:1];
        for (size_t i = 0; i < compilation_->inputs_.size(); ++i) {
//          std::unique_ptr<OperandInfo>& input_data = inputs_info_[i];
          MPSImage* mps_img = input_mpsimages_[i];
          const id<MTLBuffer> mtl_buffer = input_mtlbuffers_[i];
          int input_index = compilation_->inputs_[i];
          UploadToMPSImage(mps_img, mtl_buffer, command_buffer,
                           compilation_->values_[input_index].data);
          [image_array addObject:mps_img];
        }

        for (size_t i = 0; i < compilation_->constants_.size(); ++i) {
          uint32_t index = compilation_->constants_[i];
          if (compilation_->values_.find(index) ==
              compilation_->values_.end()) {
            std::cout << "Can't find constant " << index;
            success = false;
            break;
          }
          const ValueInfo& value_info =
              compilation_->values_[compilation_->constants_[i]];
          MPSImage* mps_img = constant_mpsimages_[i];
          const id<MTLBuffer> mtl_buffer = constant_mtlbuffers_[i];
          UploadToMPSImage(mps_img, mtl_buffer, command_buffer, value_info.data);
          [image_array addObject:mps_img];
        }

        std::map<uint32_t, MPSImage*> output_mps_images;
        std::map<uint32_t, MPSImage*> temporary_mps_images;
        EncodeCustomKernel(command_buffer, output_mps_images,
                           temporary_mps_images, compilation_->inputs_[0]);
        for (size_t i = 0; i < compilation_->graphs_.size(); i++) {
          // temporary_inputs_[i -1] is the temporary input image index.
          // temporary_mps_images[temporary_inputs_[i -1]] is temporary input
          // image.
          NSMutableArray<MPSImage*>* source_images;
          if (i == 0) {
            // image_array is First graph
            source_images = image_array;
          } else {
            source_images = [NSMutableArray arrayWithCapacity:1];
            NSArray<id<MPSHandle>>* source_image_handles =
                compilation_->graphs_[i].sourceImageHandles;
            if (source_image_handles.count) {
              // There are only one paramters for new graph.
              assert(source_image_handles.count == 1);
              uint32_t input_index = [source_image_handles[0].label intValue];
              // Find temporary input images of next graph.
              [source_images addObject:temporary_mps_images[input_index]];
            }
          }
          NSMutableArray<MPSImage*>* intermediate_images =
              [NSMutableArray arrayWithCapacity:1];

          MPSImage* graph_output_image = [compilation_->graphs_[i]
              encodeToCommandBuffer:command_buffer
                       sourceImages:source_images
                       sourceStates:nullptr
                 intermediateImages:intermediate_images
                  destinationStates:nullptr];

          SaveTemporaryImages(temporary_mps_images, intermediate_images);
          // The order of graph is not the same as compilation_->output_.
          uint32_t output_index = [graph_output_image.label intValue];
          output_mps_images[output_index] = graph_output_image;
        }

        for (size_t i = 0; i < compilation_->outputs_.size(); ++i) {
          MPSImage* output_img = output_mps_images[compilation_->outputs_[i]];
          id<MTLBuffer> output_buffer = output_mtlbuffers_[i];

          id<MTLComputeCommandEncoder> encoder =
              [command_buffer computeCommandEncoder];
          id<MTLComputePipelineState> state =
              GetMPSCNNContext().GetSpecializedPipelineState(
                  KernelFor(output_img, @"copy_metal_to_nhwc",
                            @"copy_metal_to_nhwc_nonarray"),
                  {{ushort(output_img.height), ushort(output_img.width),
                    ushort(output_img.featureChannels)}});

          [encoder setComputePipelineState:state];
          [encoder setBuffer:output_buffer offset:0 atIndex:0];
          [encoder setTexture:[output_img texture] atIndex:0];

          const auto& outputLaunchParams =
              SpatialPointwiseKernelLaunchParams(state, output_img);
          [encoder
               dispatchThreadgroups:outputLaunchParams.threadgroupsPerGrid
              threadsPerThreadgroup:outputLaunchParams.threadsPerThreadgroup];
          [encoder endEncoding];
        }

        [command_buffer commit];
        [command_buffer waitUntilCompleted];

        for (size_t i = 0; i < compilation_->outputs_.size(); ++i) {
//          std::unique_ptr<OperandInfo>& output_data = outputs_info_[i];
          int output_index = compilation_->outputs_[i];
//          ValueInfo& output_data = compilation_->values_[output_index];
          id<MTLBuffer> output_buffer = output_mtlbuffers_[i];
          const OperandMac& output_operand = compilation_->operands_[output_index];
          std::vector<float> output_data(output_operand.requiredSize() / sizeof(float));
          memcpy(output_data.data(), [output_buffer contents],
                 output_data.size() * sizeof(float));
//          for (size_t j = 0; j < output_data.size(); ++j) {
//            std::cout << " ==== " << output_data[j] << "\n";
//          }
          output_data_ = output_data;
        }
      }  // @autoreleasepool
    } while (0);
  }
}
  
std::vector<float> ExecutionImplMacMPS::OutputData() {
  return output_data_;
}

void ExecutionImplMacMPS::UploadToMPSImage(
    const MPSImage* mps_image,
    const id<MTLBuffer>& mtl_buffer,
    const id<MTLCommandBuffer>& command_buffer,
    const std::vector<float>& data) {
  if (@available(macOS 10.13, *)) {
    memcpy([mtl_buffer contents], data.data(), data.size() * sizeof(float));
    id<MTLComputeCommandEncoder> encoder =
        [command_buffer computeCommandEncoder];
    id<MTLComputePipelineState> state =
        GetMPSCNNContext().GetSpecializedPipelineState(
            KernelFor(mps_image, @"copy_nhwc_to_metal",
                      @"copy_nhwc_to_metal_nonarray"),
            {{ushort(mps_image.height), ushort(mps_image.width),
              ushort(mps_image.featureChannels)}});
    [encoder setComputePipelineState:state];
    [encoder setBuffer:mtl_buffer offset:0 atIndex:0];
    [encoder setTexture:[mps_image texture] atIndex:0];
    const auto& inputLaunchParams =
        SpatialPointwiseKernelLaunchParams(state, mps_image);
    [encoder dispatchThreadgroups:inputLaunchParams.threadgroupsPerGrid
            threadsPerThreadgroup:inputLaunchParams.threadsPerThreadgroup];
    [encoder endEncoding];
  }
}

}  // namespace ml
