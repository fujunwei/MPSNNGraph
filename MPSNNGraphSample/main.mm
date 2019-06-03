//
//  main.cpp
//  MPSNativeSample
//
//  Created by mac-webgl-stable on 1/31/19.
//  Copyright © 2019 mac-webgl-stable. All rights reserved.
//

#import <iostream>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include "mpscnn_context.h"
#include "depthwise_conv_test.h"

int main(int argc, const char * argv[]) {
//  ml::DepthwiseConv2dFloatLarge();
  // Build the graph.
  // Create placeholders for inputs.
  MPSNNImageNode* tensor0 = [MPSNNImageNode nodeWithHandle:nullptr];
  MPSNNImageNode* tensor1 = [MPSNNImageNode nodeWithHandle:nullptr];
  MPSNNImageNode* tensor2 = [MPSNNImageNode nodeWithHandle:nullptr];
  MPSNNImageNode* tensor3 = [MPSNNImageNode nodeWithHandle:nullptr];
  MPSNNAdditionNode* add_0 = [MPSNNAdditionNode nodeWithLeftSource:tensor0
                                                       rightSource:tensor1];
  MPSNNAdditionNode* add_1 = [MPSNNAdditionNode nodeWithLeftSource:tensor2
                                                       rightSource:tensor3];
  MPSNNMultiplicationNode* mul = [MPSNNMultiplicationNode nodeWithLeftSource:add_0.resultImage
                                                                 rightSource:add_1.resultImage];
  MPSNNGraph* graph = [MPSNNGraph graphWithDevice:ml::GetMPSCNNContext().device
                                      resultImage:mul.resultImage
                              resultImageIsNeeded:true];
  
  // Set inputs data.
  const std::vector<int> shape = {2, 2, 2, 2};
  size_t size = 16;
  const std::vector<float> constant_data(size, 0.5);
  id<MTLCommandBuffer> command_buffer = [ml::GetMPSCNNContext().command_queue commandBuffer];
  MPSImage* constant0 = ml::GetMPSCNNContext().CreateMPSImage(command_buffer, shape, constant_data);
  MPSImage* constant1 = ml::GetMPSCNNContext().CreateMPSImage(command_buffer, shape, constant_data);
  NSMutableArray<MPSImage*>* image_array = [NSMutableArray arrayWithCapacity:1];
  const std::vector<float> input_data0(size, 1);
  const std::vector<float> input_data1(size, 2);
  MPSImage* input0 = ml::GetMPSCNNContext().CreateMPSImage(command_buffer, shape, input_data0);
  MPSImage* input1 = ml::GetMPSCNNContext().CreateMPSImage(command_buffer, shape, input_data1);
  [image_array addObject:constant0];
  [image_array addObject:input0];
  [image_array addObject:constant1];
  [image_array addObject:input1];
  
  // Execution Graph.
  MPSImage* output_image = [graph encodeToCommandBuffer:command_buffer
                                  sourceImages:image_array
                                  sourceStates:nullptr
                                  intermediateImages:nullptr
                                  destinationStates:nullptr];
  id<MTLBuffer> output_buffer = ml::GetMPSCNNContext().OutputBuffer(command_buffer, output_image, size * sizeof(float));
  [command_buffer commit];
  [command_buffer waitUntilCompleted];
  
  // Get output data.
  std::vector<float> output_data(size);
  memcpy(output_data.data(), [output_buffer contents], output_data.size() * sizeof(float));
  std::cout << "[";
  for (size_t i = 0; i < size; ++i)
  {
    std::cout << output_data[i] << ' ';
  }
  std::cout << ']' << std::endl;

  return 0;
}
