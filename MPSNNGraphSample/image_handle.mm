// Copyright 2017 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "image_handle.h"

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
