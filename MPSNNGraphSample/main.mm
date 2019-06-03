
#import <iostream>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include "mpscnn_context.h"

int main(int argc, const char * argv[]) {
  // Build the graph.
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
  MPSNNGraph* graph = [MPSNNGraph graphWithDevice:util::DefaultDevice()
                                      resultImage:mul.resultImage
                              resultImageIsNeeded:true];
  
  // Set inputs data.
  const std::vector<int> shape = {2, 2, 2, 2};
  size_t length = 16;
  const std::vector<float> constant_data(length, 0.5);
  id<MTLCommandBuffer> command_buffer = [util::CommandQueue() commandBuffer];
  MPSImage* constant0 = util::CreateMPSImage(command_buffer, shape, constant_data);
  MPSImage* constant1 = util::CreateMPSImage(command_buffer, shape, constant_data);
  NSMutableArray<MPSImage*>* image_array = [NSMutableArray arrayWithCapacity:1];
  const std::vector<float> input_data0(length, 1);
  const std::vector<float> input_data1(length, 2);
  MPSImage* input0 = util::CreateMPSImage(command_buffer, shape, input_data0);
  MPSImage* input1 = util::CreateMPSImage(command_buffer, shape, input_data1);
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
  size_t size = length * sizeof(float);
  id<MTLBuffer> output_buffer = util::OutputBuffer(command_buffer, output_image, size);
  [command_buffer commit];
  [command_buffer waitUntilCompleted];
  
  // Get output data.
  std::vector<float> output_data(length);
  memcpy(output_data.data(), [output_buffer contents], size);
  std::cout << "[";
  for (size_t i = 0; i < length; ++i)
  {
    std::cout << output_data[i] << ' ';
  }
  std::cout << ']' << std::endl;

  return 0;
}
