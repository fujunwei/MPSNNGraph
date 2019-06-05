#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

@interface MPSImageHandle : NSObject <MPSHandle>

@property(nonatomic, copy) NSString* label_;
@property(nonatomic, retain) MPSImage* image_;

- (id)initWithImage:(MPSImage*)image;

-(MPSImage*) image;

- (id)initWithLabel:(NSString*)label;

@end

id<MTLBuffer> CreateOutputBuffer(id<MTLDevice> device,
                                 id<MTLCommandBuffer> command_buffer,
                                 const MPSImage* output_img,
                                 size_t size);
