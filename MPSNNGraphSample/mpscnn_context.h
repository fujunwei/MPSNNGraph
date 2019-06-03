
// Copyright 2017 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef SERVICES_ML_MPSCNNCONTEXT_H_
#define SERVICES_ML_MPSCNNCONTEXT_H_

#import <Metal/MTLBuffer.h>
#import <Metal/MTLDevice.h>
#import <Metal/MTLLibrary.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include <vector>
#include <string>
#include <unordered_map>

struct API_AVAILABLE(macosx(10.13)) MPSCNNContext {
 public:
  MPSCNNContext();
  ~MPSCNNContext();
  id<MTLDevice> device;
  id<MTLCommandQueue> command_queue;
  id<MTLLibrary> library;
  bool initialized;

  bool IsValid() const {
    return initialized && device != nil && library != nil;
  }

  id<MTLComputePipelineState> GetPipelineState(NSString* kernel);
  id<MTLComputePipelineState> GetSpecializedPipelineState(NSString* kernel,
                                                          const std::vector<ushort>& constants);
  
 private:
  std::unordered_map<std::string, id<MTLComputePipelineState>> pipelineCache_;
};

MPSCNNContext& API_AVAILABLE(macosx(10.13)) GetMPSCNNContext();

namespace util {

  id<MTLDevice> DefaultDevice();
  id<MTLCommandQueue> CommandQueue();
  MPSImage* CreateMPSImage(const std::vector<int>& shape);
  MPSImage* CreateMPSImageWithData(id<MTLCommandBuffer> command_buffer,
                           const std::vector<float>& data, const std::vector<int>& shape);
  void UploadDataToMPSImage(id<MTLCommandBuffer> command_buffer, MPSImage* mps_image,
                           const std::vector<float>& data);
  id<MTLBuffer> OutputBuffer(id<MTLCommandBuffer> command_buffer, const MPSImage* output_img, size_t size);

}

@interface MPSImageHandle : NSObject <MPSHandle>

@property(nonatomic, copy) NSString* label_;
@property(nonatomic, retain) MPSImage* image_;

- (id)initWithImage:(MPSImage*)image;

-(MPSImage*) image;    // return the MPSImage corresponding to the handle

- (id)initWithLabel:(NSString*)label;

@end

#endif  // SERVICES_ML_MPSCNNCONTEXT_H_
