// Copyright 2017 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "mpscnn_context.h"

#include <iostream>
#import <Metal/MTLFunctionConstantValues.h>

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
  
}

static const char* MPSCNN_KERNELS = R"V0G0N(


using namespace metal;

constant ushort ushort_arg_0[[function_constant(0)]];
constant ushort ushort_arg_1[[function_constant(1)]];
constant ushort ushort_arg_2[[function_constant(2)]];
constant ushort ushort_arg_3[[function_constant(3)]];
constant ushort ushort_arg_4[[function_constant(4)]];
constant ushort ushort_arg_5[[function_constant(5)]];
constant ushort ushort_arg_6[[function_constant(6)]];
constant ushort ushort_arg_7[[function_constant(7)]];
constant ushort ushort_arg_8[[function_constant(8)]];
constant ushort ushort_arg_9[[function_constant(9)]];

inline constexpr ushort divRoundUp(ushort x, ushort y) { return (x + (y - 1)) / y; }

kernel void copy_nhwc_to_metal(constant float* in[[buffer(0)]],
                               texture2d_array<half, access::write> out[[texture(0)]],
                               ushort3 gid[[thread_position_in_grid]]) {
    const ushort H = ushort_arg_0;
    const ushort W = ushort_arg_1;
    const ushort C = ushort_arg_2;
    if (gid.x >= W || gid.y >= H) {
        return;
    }
    
    const ushort n = gid.z / divRoundUp(C, 4);
    const ushort c = gid.z - n * divRoundUp(C, 4);
    
    // TODO: are the `else` branches needed?
    // TODO: trick the optimizer for case where C == 4?
#define HWC_TO_CHWP4(idx, n, c_, h, w)                                     \
if ((c_) < C) {                                                          \
trns[idx] = in[n * H * W * C + int(h) * W * C + int(w) * C + int(c_)]; \
} else {                                                                 \
trns[idx] = 0.0h;                                                      \
}
    
    half4 trns;
    HWC_TO_CHWP4(0, n, c * 4 + 0, gid.y, gid.x);
    HWC_TO_CHWP4(1, n, c * 4 + 1, gid.y, gid.x);
    HWC_TO_CHWP4(2, n, c * 4 + 2, gid.y, gid.x);
    HWC_TO_CHWP4(3, n, c * 4 + 3, gid.y, gid.x);
#undef HWC_TO_CHWP4
    
    out.write(trns, gid.xy, gid.z);
}

kernel void copy_nhwc_to_metal_nonarray(constant float* in[[buffer(0)]],
                                        texture2d<half, access::write> out[[texture(0)]],
                                        ushort2 gid[[thread_position_in_grid]]) {
    const ushort H = ushort_arg_0;
    const ushort W = ushort_arg_1;
    const ushort C = ushort_arg_2;
    
    if (gid.x >= W || gid.y >= H) {
        return;
    }
    
    half4 trns;
    // TODO: are the `else` branches needed?
    // TODO: trick the optimizer for case where C % 4 == 0?
    
#define HWC_TO_CHWP4(idx, c, h, w)                      \
if ((c) < C) {                                          \
  trns[idx] = in[int(h) * W * C + int(w) * C + int(c)]; \
} else {                                                \
  trns[idx] = 0.0h;                                     \
}
    
    HWC_TO_CHWP4(0, 0, gid.y, gid.x);
    HWC_TO_CHWP4(1, 1, gid.y, gid.x);
    HWC_TO_CHWP4(2, 2, gid.y, gid.x);
    HWC_TO_CHWP4(3, 3, gid.y, gid.x);
#undef HWC_TO_CHWP4
    
    out.write(trns, gid.xy);
}

kernel void copy_metal_to_nhwc(texture2d_array<half, access::read> in[[texture(0)]],
                               device float* out[[buffer(0)]],
                               ushort3 gid[[thread_position_in_grid]]) {
    const ushort H = ushort_arg_0;
    const ushort W = ushort_arg_1;
    const ushort C = ushort_arg_2;
    
    if (gid.x >= W || gid.y >= H) {
        return;
    }
    const ushort n = gid.z / divRoundUp(C, 4);
    const ushort c = gid.z - n * divRoundUp(C, 4);
    
    half4 cs = in.read(gid.xy, gid.z);
    
#define CHWP4_TO_HWC(idx, n, c_, h, w)                                  \
if ((c_) < C) {                                                         \
  out[n * H * W * C + int(h) * W * C + int(w) * C + int(c_)] = cs[idx];     \
}
    
    CHWP4_TO_HWC(0, n, c * 4 + 0, gid.y, gid.x);
    CHWP4_TO_HWC(1, n, c * 4 + 1, gid.y, gid.x);
    CHWP4_TO_HWC(2, n, c * 4 + 2, gid.y, gid.x);
    CHWP4_TO_HWC(3, n, c * 4 + 3, gid.y, gid.x);
#undef CHWP4_TO_HWC
}

kernel void copy_metal_to_nhwc_nonarray(texture2d<half, access::read> in[[texture(0)]],
                                        device float* out[[buffer(0)]],
                                        ushort2 gid[[thread_position_in_grid]]) {
    const ushort H = ushort_arg_0;
    const ushort W = ushort_arg_1;
    const ushort C = ushort_arg_2;
    
    if (gid.x >= W || gid.y >= H) {
        return;
    }
    
    half4 cs = in.read(gid.xy);
    
#define CHWP4_TO_HWC(idx, c, h, w)                       \
if ((c) < C) {                                         \
out[int(h) * W * C + int(w) * C + int(c)] = cs[idx];  \
}
    
    CHWP4_TO_HWC(0, 0, gid.y, gid.x);
    CHWP4_TO_HWC(1, 1, gid.y, gid.x);
    CHWP4_TO_HWC(2, 2, gid.y, gid.x);
    CHWP4_TO_HWC(3, 3, gid.y, gid.x);
#undef CHWP4_TO_HWC
}

)V0G0N";

MPSCNNContext::MPSCNNContext() = default;
MPSCNNContext::~MPSCNNContext() = default;

MPSCNNContext& GetMPSCNNContext() {
  static MPSCNNContext ctx;
  if (!ctx.initialized) {
    ctx.initialized = true;

    ctx.device = MTLCreateSystemDefaultDevice();
    if (ctx.device == nil) {
      std::cout << "Cannot create MTLDevice";
      return ctx;
    } else {
      std::cout << "Created MTLDevice: " << ctx.device.name.UTF8String;
    }

    NSError* compileError = nil;
    ctx.library = [ctx.device newLibraryWithSource:[NSString stringWithUTF8String:MPSCNN_KERNELS]
        options:nil
        error:&compileError];
    if (compileError != nil || ctx.library == nil) {
      std::cout << "Failed to load kernels: " << [[compileError localizedDescription] UTF8String];
      return ctx;
    }

    ctx.command_queue = [ctx.device newCommandQueue];
  };
  return ctx;
}

id<MTLComputePipelineState> MPSCNNContext::GetPipelineState(NSString* kernel) {
  std::string kernelStr = std::string([kernel UTF8String]);
  if (pipelineCache_.find(kernelStr) != pipelineCache_.end()) {
    std::cout << "Hit in pipeline cache for: " << kernelStr;
    return pipelineCache_[kernelStr];
  }
  id<MTLFunction> func = [library newFunctionWithName:kernel];
  if (!func) {
    std::cout << "Couldn't get function: " << kernelStr;
    return nullptr;
  }
  NSError* errors;
  id<MTLComputePipelineState> state =
      [device newComputePipelineStateWithFunction:func error:&errors];
  if (!state) {
    std::cout << "Couldn't get state: " << kernelStr;
    return nullptr;
  }
  pipelineCache_[kernelStr] = state;
  return state;
}

id<MTLComputePipelineState> MPSCNNContext::GetSpecializedPipelineState(
    NSString* kernel, const std::vector<ushort>& constants) {
  std::string kernelStr = std::string([kernel UTF8String]);
  for (size_t i = 0; i < constants.size(); ++i) {
    kernelStr += "_" + std::to_string(constants[i]);
  }
  if (pipelineCache_.find(kernelStr) != pipelineCache_.end()) {
    std::cout << "Hit in pipeline cache for: " << kernelStr;
    return pipelineCache_[kernelStr];
  }
  MTLFunctionConstantValues* constantValues = [MTLFunctionConstantValues new];
  for (size_t i = 0; i < constants.size(); ++i) {
    [constantValues setConstantValue:&constants[i] type:MTLDataTypeUShort atIndex:i];
  }
  NSError* errors;

  id<MTLFunction> func =
      [library newFunctionWithName:kernel constantValues:constantValues error:&errors];
  if (!func) {
    std::cout << "Couldn't get function: " <<
                kernelStr <<
                " error: " <<
                [[errors localizedDescription] UTF8String];
    return nullptr;
  }
  id<MTLComputePipelineState> state =
      [device newComputePipelineStateWithFunction:func error:&errors];
  if (!state) {
    std::cout << "Couldn't get function: " <<
                kernelStr <<
                " error: " <<
                [[errors localizedDescription] UTF8String];
    return nullptr;
  }
  pipelineCache_[kernelStr] = state;
  return state;
}

namespace util {

id<MTLDevice> DefaultDevice() {
  return GetMPSCNNContext().device;
}

id<MTLCommandQueue> CommandQueue() {
  return GetMPSCNNContext().command_queue;
}

MPSImage* CreateMPSImage(const std::vector<int>& shape) {
  // Ceate MPSImage for inputs and outputs.
  MPSImageDescriptor* image_desc = [MPSImageDescriptor
                                    imageDescriptorWithChannelFormat:MPSImageFeatureChannelFormatFloat32
                                    width:shape[2]
                                    height:shape[1]
                                    featureChannels:shape[3]
                                    numberOfImages:shape[0]
                                    usage:MTLTextureUsageShaderRead |
                                    MTLTextureUsageShaderWrite];
  MPSImage* mps_image = [[MPSImage alloc] initWithDevice:GetMPSCNNContext().device
                                         imageDescriptor:image_desc];
  return mps_image;
}

MPSImage* CreateMPSImageWithData(id<MTLCommandBuffer> command_buffer, const std::vector<float>& data,
                                 const std::vector<int>& shape) {
  // Ceate MPSImage for inputs and outputs.
  MPSImage* mps_image = CreateMPSImage(shape);
  UploadDataToMPSImage(command_buffer, mps_image, data);
  return mps_image;
}
  
void UploadDataToMPSImage(id<MTLCommandBuffer> command_buffer, MPSImage* mps_image,
                          const std::vector<float>& data) {
  size_t size = data.size() * sizeof(float);
  id<MTLBuffer> mtl_buffer = [GetMPSCNNContext().device newBufferWithLength:size
                                            options:MTLResourceOptionCPUCacheModeWriteCombined];
  //    id<MTLCommandBuffer> command_buffer = [GetMPSCNNContext().command_queue commandBuffer];
  
  memcpy([mtl_buffer contents], data.data(), data.size() * sizeof(float));
  id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
  id<MTLComputePipelineState> state =
  GetMPSCNNContext().GetSpecializedPipelineState(KernelFor(mps_image, @"copy_nhwc_to_metal",
                                                           @"copy_nhwc_to_metal_nonarray"),
                                                 {{ushort(mps_image.height), ushort(mps_image.width),
    ushort(mps_image.featureChannels)}});
  [encoder setComputePipelineState:state];
  [encoder setBuffer:mtl_buffer offset:0 atIndex:0];
  [encoder setTexture:[mps_image texture] atIndex:0];
  const auto& inputLaunchParams = SpatialPointwiseKernelLaunchParams(state, mps_image);
  [encoder dispatchThreadgroups:inputLaunchParams.threadgroupsPerGrid
          threadsPerThreadgroup:inputLaunchParams.threadsPerThreadgroup];
  [encoder endEncoding];
}

id<MTLBuffer> OutputBuffer(id<MTLCommandBuffer> command_buffer, const MPSImage* output_img, size_t size) {
  id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
  id<MTLComputePipelineState> state =
  GetMPSCNNContext().GetSpecializedPipelineState(
                                                 KernelFor(output_img, @"copy_metal_to_nhwc",
                                                           @"copy_metal_to_nhwc_nonarray"),
                                                 {{ushort(output_img.height), ushort(output_img.width),
    ushort(output_img.featureChannels)}});
  id<MTLBuffer> output_buffer = [GetMPSCNNContext().device
                                 newBufferWithLength:size
                                 options:MTLResourceOptionCPUCacheModeWriteCombined];
  
  [encoder setComputePipelineState:state];
  [encoder setBuffer:output_buffer offset:0 atIndex:0];
  [encoder setTexture:[output_img texture] atIndex:0];
  
  const auto& outputLaunchParams =
  SpatialPointwiseKernelLaunchParams(state, output_img);
  [encoder
   dispatchThreadgroups:outputLaunchParams.threadgroupsPerGrid
   threadsPerThreadgroup:outputLaunchParams.threadsPerThreadgroup];
  [encoder endEncoding];
  return output_buffer;
}

}

@implementation MPSImageHandle

@synthesize label_ = _label;
@synthesize image_ = _image;

+ (BOOL)supportsSecureCoding {
  return YES;
}

- (id)initWithCoder:(NSCoder*)coder {
  self = [super init];
  return self;
}

- (void)encodeWithCoder:(NSCoder*)aCoder {
}

- (id)initWithLabel:(NSString*)label {
  self = [super init];
  self.label_ = label;
  return self;
}

- (id)initWithImage:(MPSImage*)image {
  self = [super init];
  self.image_ = image;
  return self;
}

-(MPSImage*) image {
  return self.image_;
}

/*! @abstract   A label to be attached to associated MTLResources for this node
 *  @return     A human readable string for debugging purposes
 */
- (NSString*)label {
  return self.label_;
}

@end
