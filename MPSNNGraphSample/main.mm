
#import <iostream>
#import <vector>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include "image_handle.h"

namespace {

MPSImage* CreateMPSImage(id<MTLDevice> device, const std::vector<int>& shape) {
  // Ceate MPSImage for inputs and outputs.
  MPSImageDescriptor* image_desc = [MPSImageDescriptor
                                    imageDescriptorWithChannelFormat:MPSImageFeatureChannelFormatFloat16
                                    width:shape[2]
                                    height:shape[1]
                                    featureChannels:shape[3]
                                    numberOfImages:shape[0]
                                    usage:MTLTextureUsageShaderRead |
                                    MTLTextureUsageShaderWrite];
  MPSImage* mps_image = [[MPSImage alloc] initWithDevice:device
                                         imageDescriptor:image_desc];
  return mps_image;
}

void UploadDataToMPSImage(MPSImage* mps_image,
                          const std::vector<__fp16>& data) {
  for (size_t i = 0; i < mps_image.numberOfImages; ++i) {
    [mps_image writeBytes:data.data()
               dataLayout:MPSDataLayoutHeightxWidthxFeatureChannels
              bytesPerRow:mps_image.width * sizeof(__fp16)
                   region:MTLRegionMake2D(0, 0, mps_image.width, mps_image.height)
       featureChannelInfo:{0, mps_image.featureChannels}
               imageIndex:i];
  }
}

MPSImage* CreateMPSImageWithData(id<MTLDevice> device, const std::vector<__fp16>& data,
                                 const std::vector<int>& shape) {
  // Ceate MPSImage for inputs and outputs.
  MPSImage* mps_image = CreateMPSImage(device, shape);
  UploadDataToMPSImage(mps_image, data);
  return mps_image;
}

}

int main(int argc, const char * argv[]) {
  // Build the graph.
  const std::vector<int> shape = {1, 4, 4, 1};
  size_t length = 16;
  const std::vector<__fp16> constant_data(length, 0);
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  id<MTLCommandBuffer> command_buffer = [[device newCommandQueue] commandBuffer];
  MPSImage* constant0 = CreateMPSImageWithData(device, constant_data, shape);
  MPSNNImageNode* tensor0 = [MPSNNImageNode nodeWithHandle:[[MPSImageHandle alloc]
                                                            initWithImage:constant0]];
  MPSImage* input0 = CreateMPSImage(device, shape);
  MPSNNImageNode* tensor1 = [MPSNNImageNode nodeWithHandle:[[MPSImageHandle alloc]
                                                            initWithImage:input0]];
  MPSImage* constant1 = CreateMPSImageWithData(device, constant_data, shape);
  MPSNNImageNode* tensor2 = [MPSNNImageNode nodeWithHandle:[[MPSImageHandle alloc]
                                                            initWithImage:constant1]];
  MPSImage* input1 = CreateMPSImage(device, shape);
  MPSNNImageNode* tensor3 = [MPSNNImageNode nodeWithHandle:[[MPSImageHandle alloc]
                                                            initWithImage:input1]];
  MPSNNAdditionNode* add_0 = [MPSNNAdditionNode nodeWithLeftSource:tensor0
                                                       rightSource:tensor1];
  MPSNNAdditionNode* add_1 = [MPSNNAdditionNode nodeWithLeftSource:tensor2
                                                       rightSource:tensor3];
  MPSNNAdditionNode* mul = [MPSNNAdditionNode nodeWithLeftSource:add_0.resultImage
                                                                 rightSource:add_1.resultImage];
  MPSNNGraph* graph = [MPSNNGraph graphWithDevice:device
                                      resultImage:mul.resultImage
                              resultImageIsNeeded:true];
  
  // Execute Graph.
  NSMutableArray<MPSImage*>* image_array = [NSMutableArray arrayWithCapacity:1];
  const std::vector<__fp16> input_data0(length, 2.2334524562);
  const std::vector<__fp16> input_data1(length, 0);
  UploadDataToMPSImage(input0, input_data0);
  UploadDataToMPSImage(input1, input_data1);
  NSArray<MPSImageHandle*> * handles = graph.sourceImageHandles;
  for (size_t i = 0; i < handles.count; ++i) {
    [image_array addObject:handles[i].image];
  }
  MPSImage* output_image = [graph encodeToCommandBuffer:command_buffer
                                           sourceImages:image_array
                                           sourceStates:nullptr
                                           intermediateImages:nullptr
                                           destinationStates:nullptr];
  size_t size = length * sizeof(__fp16);
  id<MTLBuffer> output_buffer = CreateOutputBuffer(device, command_buffer, output_image, size);

  [command_buffer commit];
  [command_buffer waitUntilCompleted];
  // Get output data.
  std::vector<__fp16> output_data(length);
  memcpy(output_data.data(), [output_buffer contents], size);
  std::cout << "[";
  for (size_t i = 0; i < length; ++i) {
    std::cout << output_data[i] << ' ';
  }
  std::cout << ']' << std::endl;

  return 0;
}
