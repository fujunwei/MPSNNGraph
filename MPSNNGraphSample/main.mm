
#import <iostream>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include "mpscnn_context.h"

int main(int argc, const char * argv[]) {
  // Build the graph.
  const std::vector<int> shape = {2, 2, 2, 2};
  size_t length = 16;
  const std::vector<float> constant_data(length, 0.5);
  id<MTLCommandBuffer> command_buffer = [util::CommandQueue() commandBuffer];
  MPSImage* constant0 = util::CreateMPSImageWithData(command_buffer, constant_data, shape);
  MPSNNImageNode* tensor0 = [MPSNNImageNode nodeWithHandle:[[MPSImageHandle alloc]
                                                            initWithImage:constant0]];
  MPSImage* input0 = util::CreateMPSImage(shape);
  MPSNNImageNode* tensor1 = [MPSNNImageNode nodeWithHandle:[[MPSImageHandle alloc]
                                                            initWithImage:input0]];
  MPSImage* constant1 = util::CreateMPSImageWithData(command_buffer, constant_data, shape);
  MPSNNImageNode* tensor2 = [MPSNNImageNode nodeWithHandle:[[MPSImageHandle alloc]
                                                            initWithImage:constant1]];
  MPSImage* input1 = util::CreateMPSImage(shape);
  MPSNNImageNode* tensor3 = [MPSNNImageNode nodeWithHandle:[[MPSImageHandle alloc]
                                                            initWithImage:input1]];
  MPSNNAdditionNode* add_0 = [MPSNNAdditionNode nodeWithLeftSource:tensor0
                                                       rightSource:tensor1];
  MPSNNAdditionNode* add_1 = [MPSNNAdditionNode nodeWithLeftSource:tensor2
                                                       rightSource:tensor3];
  MPSNNMultiplicationNode* mul = [MPSNNMultiplicationNode nodeWithLeftSource:add_0.resultImage
                                                                 rightSource:add_1.resultImage];
  MPSNNGraph* graph = [MPSNNGraph graphWithDevice:util::DefaultDevice()
                                      resultImage:mul.resultImage
                              resultImageIsNeeded:true];
  
  
  
  // Execution Graph.
  NSMutableArray<MPSImage*>* image_array = [NSMutableArray arrayWithCapacity:1];
  const std::vector<float> input_data0(length, 1);
  const std::vector<float> input_data1(length, 2);
  util::UploadDataToMPSImage(command_buffer, input0, input_data0);
  util::UploadDataToMPSImage(command_buffer, input1, input_data1);
  NSArray<MPSImageHandle*> * handles = graph.sourceImageHandles;
  for (size_t i = 0; i < handles.count; ++i) {
    [image_array addObject:handles[i].image];
  }
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
