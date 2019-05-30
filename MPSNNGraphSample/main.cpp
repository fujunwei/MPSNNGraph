//
//  main.cpp
//  MPSNativeSample
//
//  Created by mac-webgl-stable on 1/31/19.
//  Copyright © 2019 mac-webgl-stable. All rights reserved.
//

#include <iostream>
#include "test_cases.h"
#include "depthwise_conv_test.h"
#include "resize_bilinear_test.h"

int main(int argc, const char * argv[]) {
  ml::ResizeBilinear65_65To513_513();
  
  ml::ResizeBilinear65_65_21To513_513_21();
  
  return 0;
}
