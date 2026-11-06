/*
 * Test-only, bounded ANE workload using the model shipped with Vision.
 * No model assets, camera access, installation, or elevated privileges.
 *
 * Device assignment is explicit and checked again after configuration:
 * https://developer.apple.com/documentation/vision/vnrequest/supportedcomputestagedevicesandreturnerror:
 * https://developer.apple.com/documentation/vision/vnrequest/setcomputedevice:forcomputestage:
 * https://developer.apple.com/documentation/vision/vnrequest/computedeviceforcomputestage:
 * This proves the configured main-stage device, not per-operation execution or
 * per-process energy attribution. The test separately requires positive power.
 */
#import <Foundation/Foundation.h>
#import <Vision/Vision.h>
#import <CoreML/CoreML.h>
#import <CoreGraphics/CoreGraphics.h>
#include <stdint.h>
#include <stdio.h>
#include <unistd.h>

static const size_t ImageSize = 512;
static const double LoadSeconds = 6.0;

static uint64_t WallMilliseconds(void) {
  return (uint64_t)(NSDate.date.timeIntervalSince1970 * 1000.0);
}

static NSError *ProbeError(NSString *message) {
  return [NSError errorWithDomain:@"AlloyANETest" code:1 userInfo:@{NSLocalizedDescriptionKey:message}];
}

static BOOL Emit(NSDictionary *value) {
  NSError *error = nil;
  NSData *data = [NSJSONSerialization dataWithJSONObject:value options:0 error:&error];
  if (!data || data.length > 4096) return NO;
  return fwrite(data.bytes, 1, data.length, stdout) == data.length
    && fputc('\n', stdout) != EOF && fflush(stdout) == 0;
}

static VNRequest *SelectRequest(MLNeuralEngineComputeDevice **selected, NSError **error) {
  NSMutableArray<NSString *> *observed = [NSMutableArray array];
  for (Class requestClass in @[[VNClassifyImageRequest class], [VNGenerateImageFeaturePrintRequest class]]) {
    VNRequest *request = [[requestClass alloc] init];
    NSDictionary<NSString *, NSArray<id<MLComputeDeviceProtocol>> *> *stages =
      [request supportedComputeStageDevicesAndReturnError:error];
    if (!stages) return nil;
    NSMutableArray<NSString *> *names = [NSMutableArray array];
    for (id<MLComputeDeviceProtocol> device in stages[VNComputeStageMain]) {
      [names addObject:NSStringFromClass([(id)device class])];
      if (![(id)device isKindOfClass:[MLNeuralEngineComputeDevice class]]) continue;
      [request setComputeDevice:device forComputeStage:VNComputeStageMain];
      id<MLComputeDeviceProtocol> assigned = [request computeDeviceForComputeStage:VNComputeStageMain];
      if (![(id)assigned isKindOfClass:[MLNeuralEngineComputeDevice class]]) {
        *error = ProbeError(@"Vision did not retain the explicit Neural Engine main-stage assignment.");
        return nil;
      }
      *selected = (MLNeuralEngineComputeDevice *)assigned;
      return request;
    }
    [observed addObject:[NSString stringWithFormat:@"%@: %@", NSStringFromClass(requestClass), names]];
  }
  // Only a difference in supported devices permits trying the second request.
  // Never replace a failed or unsupported ANE execution with CPU/GPU execution.
  *error = ProbeError([NSString stringWithFormat:@"No candidate supports ANE for its main stage: %@", observed]);
  return nil;
}

static BOOL PerformImage(VNRequest *request, CGContextRef bitmap, uint64_t frame, NSError **error) {
  uint8_t *pixels = CGBitmapContextGetData(bitmap);
  if (!pixels) { *error = ProbeError(@"Could not access the synthetic image bitmap."); return NO; }
  const size_t stride = CGBitmapContextGetBytesPerRow(bitmap);
  for (size_t y = 0; y < ImageSize; y++) {
    uint8_t *row = pixels + y * stride;
    for (size_t x = 0; x < ImageSize; x++) {
      // Vary the RGB input each time. Vision discards alpha, so keep it opaque.
      row[4*x] = (uint8_t)(3*x + y + 17*frame);
      row[4*x+1] = (uint8_t)(x ^ (5*y) ^ (31*frame));
      row[4*x+2] = ((x/16 + y/16 + frame) & 1) ? 192 : 32;
      row[4*x+3] = 255;
    }
  }
  CGImageRef image = CGBitmapContextCreateImage(bitmap);
  if (!image) { *error = ProbeError(@"Could not create the synthetic input image."); return NO; }
  @try {
    VNImageRequestHandler *handler = [[VNImageRequestHandler alloc] initWithCGImage:image options:@{}];
    if (![handler performRequests:@[request] error:error]) return NO;
    if (!request.results.count) { *error = ProbeError(@"The selected Vision request produced no observations."); return NO; }
    return YES;
  } @finally {
    CGImageRelease(image);
  }
}

static int Run(void) {
  if (geteuid() == 0) {
    fputs("ANE acceptance must run as a normal user.\n", stderr);
    return 1;
  }
  NSError *error = nil;
  MLNeuralEngineComputeDevice *device = nil;
  VNRequest *request = SelectRequest(&device, &error);
  if (!request) { fprintf(stderr, "%s\n", error.localizedDescription.UTF8String); return 1; }

  CGColorSpaceRef colors = CGColorSpaceCreateDeviceRGB();
  if (!colors) { fputs("Could not create the image color space.\n", stderr); return 1; }
  CGContextRef bitmap = CGBitmapContextCreate(NULL, ImageSize, ImageSize, 8, ImageSize*4, colors,
    kCGImageAlphaPremultipliedLast | kCGBitmapByteOrder32Big);
  CGColorSpaceRelease(colors);
  if (!bitmap) { fputs("Could not allocate the synthetic image bitmap.\n", stderr); return 1; }
  @try {
    @autoreleasepool {
      if (!PerformImage(request, bitmap, 0, &error)) {
        fprintf(stderr, "ANE warmup failed: %s\n", error.localizedDescription.UTF8String);
        return 1;
      }
    }
    const double started = NSProcessInfo.processInfo.systemUptime;
    if (!Emit(@{@"event":@"ready", @"at_unix_ms":@(WallMilliseconds()), @"completed_requests":@0,
      @"effective_uid":@(geteuid()), @"pid":@(getpid()), @"request":NSStringFromClass(request.class),
      @"revision":@(request.revision), @"selected_device":@"MLNeuralEngineComputeDevice",
      @"device_class":NSStringFromClass(device.class), @"neural_engine_cores":@(device.totalCoreCount),
      @"load_duration_ms":@(LoadSeconds*1000), @"warmup_completed_requests":@1})) return 1;
    uint64_t completed = 0;
    double reported = started;
    while (NSProcessInfo.processInfo.systemUptime - started < LoadSeconds && completed < 100000) {
      @autoreleasepool {
        if (!PerformImage(request, bitmap, completed+1, &error)) {
          fprintf(stderr, "ANE workload failed: %s\n", error.localizedDescription.UTF8String);
          return 1;
        }
        completed++;
        const double now = NSProcessInfo.processInfo.systemUptime;
        if (now - reported >= 0.2) {
          if (!Emit(@{@"event":@"progress", @"at_unix_ms":@(WallMilliseconds()), @"completed_requests":@(completed)})) return 1;
          reported = now;
        }
      }
    }
    if (!completed) { fputs("No ANE requests completed during the measured workload.\n", stderr); return 1; }
    return Emit(@{@"event":@"complete", @"at_unix_ms":@(WallMilliseconds()), @"completed_requests":@(completed),
      @"duration_ms":@((NSProcessInfo.processInfo.systemUptime-started)*1000)}) ? 0 : 1;
  } @finally {
    CGContextRelease(bitmap);
  }
}

int main(int argc, const char *argv[]) {
  (void)argv;
  if (argc != 1) { fputs("This bounded test fixture takes no arguments.\n", stderr); return 64; }
  @autoreleasepool {
    if (@available(macOS 14.0, *)) {
      @try { return Run(); }
      @catch (NSException *exception) { fprintf(stderr, "ANE workload exception: %s\n", exception.reason.UTF8String); return 1; }
    }
    fputs("ANE stage selection requires macOS 14 or newer.\n", stderr);
    return 1;
  }
}
