//
//  constants.h
//  MPSNativeSample
//
//  Created by mac-webgl-stable on 1/31/19.
//  Copyright © 2019 mac-webgl-stable. All rights reserved.
//

#ifndef constants_h
#define constants_h

// Operand types.
const int32_t FLOAT32 = 0;
const int32_t INT32 = 1;
const int32_t UINT32 = 2;
const int32_t TENSOR_FLOAT32 = 3;
const int32_t TENSOR_INT32 = 4;
const int32_t TENSOR_QUANT8_ASYMM = 5;

// Operation types.
const int32_t ADD = 0;
const int32_t AVERAGE_POOL_2D = 1;
const int32_t CONCATENATION = 2;
const int32_t CONV_2D = 3;
const int32_t DEPTHWISE_CONV_2D = 4;
const int32_t DEPTH_TO_SPACE = 5;
const int32_t DEQUANTIZE = 6;
const int32_t EMBEDDING_LOOKUP = 7;
const int32_t FLOOR = 8;
const int32_t FULLY_CONNECTED = 9;
const int32_t HASHTABLE_LOOKUP = 10;
const int32_t L2_NORMALIZATION = 11;
const int32_t L2_POOL_2D = 12;
const int32_t LOCAL_RESPONSE_NORMALIZATION = 13;
const int32_t LOGISTIC = 14;
const int32_t LSH_PROJECTION = 15;
const int32_t LSTM = 16;
const int32_t MAX_POOL_2D = 17;
const int32_t MUL = 18;
const int32_t RELU = 19;
const int32_t RELU1 = 20;
const int32_t RELU6 = 21;
const int32_t RESHAPE = 22;
const int32_t RESIZE_BILINEAR = 23;
const int32_t RNN = 24;
const int32_t SOFTMAX = 25;
const int32_t SPACE_TO_DEPTH = 26;
const int32_t SVDF = 27;
const int32_t TANH = 28;
const int32_t ATROUS_CONV_2D = 10003;
const int32_t ATROUS_DEPTHWISE_CONV_2D = 10004;

// Fused activation function types.
const int32_t FUSED_NONE = 0;
const int32_t FUSED_RELU = 1;
const int32_t FUSED_RELU1 = 2;
const int32_t FUSED_RELU6 = 3;

// Implicit padding algorithms.
const int32_t PADDING_SAME = 1;
const int32_t PADDING_VALID = 2;

// Execution preferences.
const int32_t PREFER_LOW_POWER = 0;
const int32_t PREFER_FAST_SINGLE_ANSWER = 1;
const int32_t PREFER_SUSTAINED_SPEED = 2;

const int32_t NOT_ERROR = 0;
const int32_t OUT_OF_MEMORY = 1;
const int32_t INCOMPLETE = 2;
const int32_t UNEXPECTED_NULL = 3;
const int32_t BAD_DATA = 4;
const int32_t OP_FAILED = 5;
const int32_t UNMAPPABLE = 5;
const int32_t BAD_STATE = 6;

#endif /* constants_h */
