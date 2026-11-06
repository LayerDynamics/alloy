/*
 * Alloy's finite, unprivileged Apple Silicon telemetry adapter.
 * No subprocesses, privileged helper, writes to hardware, or elevation path.
 * IOReport is a private, driver-dependent API; unavailable values remain null.
 *
 * IOReport ABI, DVFS encodings and macOS 27 CLPC identifiers were checked against
 * macmon 880f641ed67628ae92713928cd6806afa3c51d04 (sources.rs, clpc.rs), and
 * Stats 482da327d4b19d54a55dfc34b2a316f359e68f41 (Modules/Sensors/bridge.h).
 * Frequency-state averages can reflect requested DVFS states rather than the
 * throttled hardware clock. They are not a proof that throttling is absent.
 *
 * The CLPC identifier catalog is adapted from macmon under its MIT license:
 * Copyright (c) 2024 vladkens
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#import <Foundation/Foundation.h>
#import <IOKit/IOKitLib.h>
#include <dlfcn.h>
#include <errno.h>
#include <math.h>
#include <net/if.h>
#include <net/if_types.h>
#include <net/route.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/sysctl.h>
#include <time.h>
#include <unistd.h>

typedef struct {
  void *library;
  CFDictionaryRef (*copyAll)(uint64_t, uint64_t);
  void *(*subscribe)(void *, CFMutableDictionaryRef, CFMutableDictionaryRef *, uint64_t, CFTypeRef);
  CFDictionaryRef (*sample)(void *, CFMutableDictionaryRef, CFTypeRef);
  CFDictionaryRef (*delta)(CFDictionaryRef, CFDictionaryRef, CFTypeRef);
  CFStringRef (*group)(CFDictionaryRef);
  CFStringRef (*subgroup)(CFDictionaryRef);
  CFStringRef (*channel)(CFDictionaryRef);
  CFStringRef (*unit)(CFDictionaryRef);
  uint64_t (*channelID)(CFDictionaryRef);
  uint64_t (*driverID)(CFDictionaryRef);
  int64_t (*integer)(CFDictionaryRef, int32_t);
  int32_t (*stateCount)(CFDictionaryRef);
  CFStringRef (*stateName)(CFDictionaryRef, int32_t);
  int64_t (*residency)(CFDictionaryRef, int32_t);
} ReportAPI;

static id Null(void) { return [NSNull null]; }
static NSString *Text(id value) { return [value isKindOfClass:[NSString class]] ? value : nil; }
static NSDictionary *Dict(id value) { return [value isKindOfClass:[NSDictionary class]] ? value : nil; }
static NSArray *Array(id value) { return [value isKindOfClass:[NSArray class]] ? value : nil; }
static NSNumber *Number(id value) {
  if (![value isKindOfClass:[NSNumber class]] || CFGetTypeID((__bridge CFTypeRef)value) == CFBooleanGetTypeID()) return nil;
  return isfinite([value doubleValue]) ? value : nil;
}
static NSString *CFText(CFStringRef value) { return value ? (__bridge NSString *)value : @""; }
static void Warn(NSMutableArray<NSString *> *warnings, NSString *message) {
  if (warnings.count < 24 && ![warnings containsObject:message]) [warnings addObject:message];
}
static double Monotonic(void) {
  struct timespec value;
  return clock_gettime(CLOCK_MONOTONIC, &value) == 0 ? (double)value.tv_sec + (double)value.tv_nsec / 1e9 : NAN;
}
static BOOL Fail(NSError **error, NSString *message) {
  if (error) *error = [NSError errorWithDomain:@"AlloyMetrics" code:1 userInfo:@{NSLocalizedDescriptionKey:message}];
  return NO;
}
static BOOL LoadAPI(ReportAPI *api, NSError **error) {
  memset(api, 0, sizeof(*api));
  api->library = dlopen("/usr/lib/libIOReport.dylib", RTLD_NOW | RTLD_LOCAL);
  if (!api->library) return Fail(error, @"The IOReport library is unavailable on this macOS installation.");
#define LOAD(member, symbol) do { \
  api->member = (__typeof__(api->member))dlsym(api->library, symbol); \
  if (!api->member) return Fail(error, [NSString stringWithFormat:@"IOReport symbol %s is unavailable.", symbol]); \
} while (0)
  LOAD(copyAll, "IOReportCopyAllChannels");
  LOAD(subscribe, "IOReportCreateSubscription");
  LOAD(sample, "IOReportCreateSamples");
  LOAD(delta, "IOReportCreateSamplesDelta");
  LOAD(group, "IOReportChannelGetGroup");
  LOAD(subgroup, "IOReportChannelGetSubGroup");
  LOAD(channel, "IOReportChannelGetChannelName");
  LOAD(unit, "IOReportChannelGetUnitLabel");
  LOAD(channelID, "IOReportChannelGetChannelID");
  LOAD(driverID, "IOReportChannelGetDriverID");
  LOAD(integer, "IOReportSimpleGetIntegerValue");
  LOAD(stateCount, "IOReportStateGetCount");
  LOAD(stateName, "IOReportStateGetNameForIndex");
  LOAD(residency, "IOReportStateGetResidency");
#undef LOAD
  return YES;
}

static NSNumber *Milliwatts(int64_t delta, NSString *unit, double seconds) {
  if (delta < 0 || !isfinite(seconds) || seconds <= 0) return nil;
  double scale = [unit isEqualToString:@"J"] ? 1000.0 : [unit isEqualToString:@"mJ"] ? 1.0
    : ([unit isEqualToString:@"uJ"] || [unit isEqualToString:@"µJ"]) ? 0.001
    : [unit isEqualToString:@"nJ"] ? 0.000001 : NAN;
  double value = (double)delta * scale / seconds;
  return isfinite(value) && value >= 0 ? @(value) : nil;
}

static NSString *Component(NSString *name) {
  if ([name hasSuffix:@"CPU Energy"]) return @"cpu";
  if ([name hasSuffix:@"GPU Energy"]) return @"gpu";
  if ([name hasPrefix:@"ANE"]) return @"ane";
  return nil;
}
static BOOL Wanted(NSString *group, NSString *subgroup, NSString *channel, NSString *unit) {
  if ([group isEqualToString:@"Energy Model"]) return Component(channel) != nil;
  if ([group isEqualToString:@"CLPC"] && [subgroup isEqualToString:@"Energy Counters"]) {
    return [@[@"CPU Energy", @"GPU Energy", @"ANE"] containsObject:channel] && [unit isEqualToString:@"nJ"];
  }
  if ([group isEqualToString:@"PMP"] && [subgroup isEqualToString:@"Energy Counters"]) {
    return [channel hasPrefix:@"ANE"] && [@[@"mJ", @"uJ", @"nJ"] containsObject:unit];
  }
  if ([group isEqualToString:@"CPU Stats"] && [subgroup isEqualToString:@"CPU Core Performance States"]) {
    return [channel containsString:@"ECPU"] || [channel containsString:@"PCPU"];
  }
  return [group isEqualToString:@"GPU Stats"] && [subgroup isEqualToString:@"GPU Performance States"] && [channel isEqualToString:@"GPUPH"];
}

typedef struct { const char *bundle; uint32_t cpu, gpu, ane; } ClpcKeys;
static const ClpcKeys CLPC[] = {
  {"com.apple.driver.AppleT6000CLPCv3",0x47479059,0x4519bf1e,0x9aaa17f8},
  {"com.apple.driver.AppleT6002CLPC",0xb501816b,0x9e2c3e8b,0x14d5a574},
  {"com.apple.driver.AppleT6020CLPC",0xe301cd75,0x8f34cc7a,0x7415fb0f},
  {"com.apple.driver.AppleT6022CLPC",0x76f5fd21,0x90624499,0x3c097307},
  {"com.apple.driver.AppleT6030CLPC",0xbe03a01a,0x472d6c9c,0xb8990c38},
  {"com.apple.driver.AppleT6031CLPC",0xd81e0084,0x1085f98b,0x313a3be1},
  {"com.apple.driver.AppleT6032CLPC",0xf4a21308,0x85e80513,0x63dc8e25},
  {"com.apple.driver.AppleT6041CLPC",0x263cf1f0,0x22e4cf59,0x7849fb5c},
  {"com.apple.driver.AppleT6050CLPC",0xd3ab60bb,0x27cd4ce5,0x8c0f1c59},
  {"com.apple.driver.AppleT6050dCLPC",0x18b3010e,0xedeca11a,0x188bd883},
  {"com.apple.driver.AppleT8103CLPCv3",0x2cdd31f2,0x21becf42,0x0667fb81},
  {"com.apple.driver.AppleT8112CLPC",0xb5435137,0x638d9d52,0xea089c36},
  {"com.apple.driver.AppleT8122CLPC",0x54b51c99,0xce2dd9dc,0x960afe2f},
  {"com.apple.driver.AppleT8132CLPC",0x4a854d94,0x3b3e3b79,0x1ff4a9ec},
  {"com.apple.driver.AppleT8140CLPC",0x9bf54436,0xef70589c,0x569d941b},
  {"com.apple.driver.AppleT8142CLPC",0x43708744,0x6a010933,0xbc5efdf0},
  {"com.apple.driver.AppleT8152CLPC",0x485d4bfc,0x11784654,0x9380c6bb},
};

static NSDictionary *Properties(io_registry_entry_t entry) {
  CFMutableDictionaryRef properties = NULL;
  if (IORegistryEntryCreateCFProperties(entry, &properties, kCFAllocatorDefault, 0) != KERN_SUCCESS) return nil;
  return CFBridgingRelease(properties);
}
static void AddCLPC(NSMutableArray *selected, ReportAPI *api) {
  // These identifiers are confirmed for macOS 27. Never apply them to another OS major.
  if ([NSProcessInfo processInfo].operatingSystemVersion.majorVersion != 27) return;
  io_iterator_t iterator = 0;
  if (IOServiceGetMatchingServices(kIOMainPortDefault, IOServiceMatching("AppleCLPC"), &iterator) != KERN_SUCCESS) return;
  io_registry_entry_t entry;
  unsigned count = 0;
  while ((entry = IOIteratorNext(iterator))) {
    uint64_t driver = 0;
    NSDictionary *properties = Properties(entry);
    kern_return_t rc = IORegistryEntryGetRegistryEntryID(entry, &driver);
    IOObjectRelease(entry);
    if (++count > 64) break;
    if (rc != KERN_SUCCESS) continue;
    NSString *bundle = Text(properties[@"CFBundleIdentifier"]);
    for (size_t i = 0; i < sizeof(CLPC)/sizeof(CLPC[0]); i++) {
      if (![bundle isEqualToString:[NSString stringWithUTF8String:CLPC[i].bundle]]) continue;
      uint64_t ids[] = {((uint64_t)16 << 32) | CLPC[i].cpu, ((uint64_t)25 << 32) | CLPC[i].gpu, ((uint64_t)24 << 32) | CLPC[i].ane};
      NSArray *names = @[@"CPU Energy", @"GPU Energy", @"ANE"];
      for (unsigned n = 0; n < 3; n++) {
        BOOL exists = NO;
        for (NSDictionary *item in selected) {
          CFDictionaryRef value = (__bridge CFDictionaryRef)item;
          if (api->driverID(value) == driver && (uint32_t)api->channelID(value) == (uint32_t)ids[n]) { exists = YES; break; }
        }
        if (exists) continue;
        uint64_t type = 1ULL | (2ULL << 16) | (1ULL << 32);
        uint64_t unit = (3ULL << 56) | (118ULL << 32); // IOReport's nJ unit descriptor.
        [selected addObject:[@{
          @"DriverID":@(driver), @"DriverName":@"AppleCLPC", @"IOReportGroupName":@"CLPC",
          @"IOReportSubGroupName":@"Energy Counters", @"IOReportChannelInfo":@{@"IOReportChannelUnit":@(unit)},
          @"LegendChannel":@[@(ids[n]), @(type), names[n]],
        } mutableCopy]];
      }
      break;
    }
  }
  IOObjectRelease(iterator);
}

static NSDictionary *RequestedChannels(ReportAPI *api, NSError **error) {
  CFDictionaryRef raw = api->copyAll(0, 0);
  if (!raw) { Fail(error, @"IOReport did not return a channel catalog."); return nil; }
  NSDictionary *all = CFBridgingRelease(raw);
  NSArray *channels = Array(all[@"IOReportChannels"]);
  if (!channels || channels.count > 50000) { Fail(error, @"Invalid or oversized IOReport channel catalog."); return nil; }
  NSMutableArray *selected = [NSMutableArray array];
  for (id value in channels) {
    NSDictionary *item = Dict(value);
    if (!item) continue;
    CFDictionaryRef cf = (__bridge CFDictionaryRef)item;
    if (Wanted(CFText(api->group(cf)), CFText(api->subgroup(cf)), CFText(api->channel(cf)), CFText(api->unit(cf)))) [selected addObject:[item mutableCopy]];
  }
  AddCLPC(selected, api);
  if (selected.count == 0) { Fail(error, @"No supported IOReport energy or performance-state channels were found."); return nil; }
  NSMutableDictionary *request = [all mutableCopy];
  request[@"IOReportChannels"] = selected;
  return request;
}

static NSString *SysctlText(const char *name) {
  size_t size = 0;
  if (sysctlbyname(name, NULL, &size, NULL, 0) != 0 || size == 0 || size > 4096) return nil;
  char buffer[4096];
  if (sysctlbyname(name, buffer, &size, NULL, 0) != 0) return nil;
  size_t length = strnlen(buffer, size);
  return [[NSString alloc] initWithBytes:buffer length:length encoding:NSUTF8StringEncoding];
}
static NSArray<NSNumber *> *Frequencies(id raw, double divisor) {
  if (![raw isKindOfClass:[NSData class]]) return nil;
  NSData *data = raw;
  if (data.length == 0 || data.length % 8 != 0 || data.length > 2048) return nil;
  const uint8_t *bytes = data.bytes;
  NSMutableArray *values = [NSMutableArray array];
  for (NSUInteger i = 0; i < data.length; i += 8) {
    uint32_t frequency = (uint32_t)bytes[i] | ((uint32_t)bytes[i+1] << 8) | ((uint32_t)bytes[i+2] << 16) | ((uint32_t)bytes[i+3] << 24);
    double mhz = frequency / divisor;
    if (!isfinite(mhz) || mhz < 0 || mhz > 20000) return nil;
    [values addObject:@(mhz)];
  }
  return values;
}
static NSDictionary *FrequencyTables(void) {
  NSMutableDictionary *tables = [NSMutableDictionary dictionary];
  NSString *chip = SysctlText("machdep.cpu.brand_string");
  BOOL hz = [chip containsString:@"M1"] || [chip containsString:@"M2"] || [chip containsString:@"M3"] || [chip containsString:@"A1"];
  double cpuDivisor = !chip.length ? NAN : hz ? 1e6 : 1e3;
  io_iterator_t iterator = 0;
  if (IOServiceGetMatchingServices(kIOMainPortDefault, IOServiceMatching("AppleARMIODevice"), &iterator) != KERN_SUCCESS) return tables;
  io_registry_entry_t entry;
  unsigned count = 0;
  while ((entry = IOIteratorNext(iterator))) {
    io_name_t rawName = {0};
    kern_return_t rc = IORegistryEntryGetName(entry, rawName);
    NSString *name = rc == KERN_SUCCESS ? [NSString stringWithUTF8String:rawName] : nil;
    NSDictionary *properties = ([name isEqualToString:@"pmgr"] || [name isEqualToString:@"pmgr-child"]) ? Properties(entry) : nil;
    IOObjectRelease(entry);
    if (++count > 4096) break;
    if (!properties) continue;
    NSArray *ecpu = Frequencies(properties[@"voltage-states1-sram"], cpuDivisor);
    NSArray *pcpu = Frequencies(properties[@"voltage-states5-sram"], cpuDivisor);
    NSArray *gpu = Frequencies(properties[@"voltage-states9"], 1e6);
    if (!tables[@"ECPU"] && ecpu) tables[@"ECPU"] = ecpu;
    if (!tables[@"PCPU"] && pcpu) tables[@"PCPU"] = pcpu;
    if (!tables[@"GPU"] && gpu) tables[@"GPU"] = gpu;
  }
  IOObjectRelease(iterator);
  return tables;
}

static NSDictionary *StateMetrics(NSArray<NSDictionary *> *states, NSArray<NSNumber *> *table, BOOL gpu) {
  double total = 0, active = 0, weighted = 0;
  NSMutableArray<NSNumber *> *residencies = [NSMutableArray array];
  if (states.count == 0 || states.count > 256) return nil;
  for (NSDictionary *state in states) {
    NSNumber *number = Number(state[@"residency"]);
    NSString *name = Text(state[@"name"]);
    if (!number || !name.length || number.doubleValue < 0) return nil;
    double value = number.doubleValue;
    total += value;
    if (![name isEqualToString:@"IDLE"] && ![name isEqualToString:@"DOWN"] && ![name isEqualToString:@"OFF"]) {
      active += value;
      [residencies addObject:number];
    }
  }
  if (!isfinite(total) || total <= 0 || !isfinite(active) || active > total) return nil;
  // Some GPU tables include an initial OFF entry; that entry has no active state.
  if (gpu && table.count == residencies.count + 1) table = [table subarrayWithRange:NSMakeRange(1, table.count - 1)];
  BOOL matched = table.count > 0 && table.count == residencies.count;
  if (matched) {
    for (NSUInteger i = 0; i < table.count; i++) {
      NSNumber *frequency = Number(table[i]);
      if (!frequency || frequency.doubleValue <= 0 || frequency.doubleValue > 20000) { matched = NO; break; }
      weighted += residencies[i].doubleValue * frequency.doubleValue;
    }
  }
  return @{
    @"total":@(total), @"active":@(active), @"active_percent":@(100.0 * active / total),
    @"weighted":matched && isfinite(weighted) ? @(weighted) : Null(),
    @"frequency_mhz":matched && active > 0 && isfinite(weighted) ? @(weighted / active) : Null(),
  };
}
static NSArray *ReadStates(ReportAPI *api, CFDictionaryRef item) {
  int32_t count = api->stateCount(item);
  if (count <= 0 || count > 256) return nil;
  NSMutableArray *states = [NSMutableArray array];
  for (int32_t i = 0; i < count; i++) [states addObject:@{@"name":CFText(api->stateName(item, i)), @"residency":@(api->residency(item, i))}];
  return states;
}

static NSDictionary *DiskCounters(void) {
  io_iterator_t iterator = 0;
  if (IOServiceGetMatchingServices(kIOMainPortDefault, IOServiceMatching("IOBlockStorageDriver"), &iterator) != KERN_SUCCESS) return nil;
  NSMutableDictionary *result = [NSMutableDictionary dictionary];
  io_registry_entry_t entry;
  unsigned count = 0;
  while ((entry = IOIteratorNext(iterator))) {
    uint64_t registryID = 0;
    kern_return_t rc = IORegistryEntryGetRegistryEntryID(entry, &registryID);
    CFTypeRef raw = IORegistryEntryCreateCFProperty(entry, CFSTR("Statistics"), kCFAllocatorDefault, 0);
    NSDictionary *statistics = Dict(CFBridgingRelease(raw));
    IOObjectRelease(entry);
    if (++count > 256) { IOObjectRelease(iterator); return nil; }
    NSNumber *read = Number(statistics[@"Bytes (Read)"]), *write = Number(statistics[@"Bytes (Write)"]);
    if (rc == KERN_SUCCESS && read && write && read.doubleValue >= 0 && write.doubleValue >= 0) {
      result[[NSString stringWithFormat:@"%llu", (unsigned long long)registryID]] = @[@(read.unsignedLongLongValue), @(write.unsignedLongLongValue)];
    }
  }
  IOObjectRelease(iterator);
  return result.count ? result : nil;
}
static NSDictionary *NetworkCounters(void) {
  // NET_RT_IFLIST2 supplies 64-bit byte counters, avoiding the 32-bit getifaddrs wrap.
  int mib[] = {CTL_NET, PF_ROUTE, 0, 0, NET_RT_IFLIST2, 0};
  size_t size = 0;
  if (sysctl(mib, 6, NULL, &size, NULL, 0) != 0 || size == 0 || size > 1024 * 1024) return nil;
  NSMutableData *buffer = [NSMutableData dataWithLength:size];
  if (sysctl(mib, 6, buffer.mutableBytes, &size, NULL, 0) != 0 || size > buffer.length) return nil;
  const uint8_t *bytes = buffer.bytes;
  NSMutableDictionary *result = [NSMutableDictionary dictionary];
  size_t offset = 0;
  // Routing dumps also contain short address messages. Read the common header
  // before requiring the larger RTM_IFINFO2 payload, then skip other types.
  struct RouteHeader { uint16_t length; uint8_t version, type; };
  while (offset + sizeof(struct RouteHeader) <= size) {
    struct RouteHeader header;
    memcpy(&header, bytes + offset, sizeof(header));
    if (header.length < sizeof(header) || header.length > size - offset) return nil;
    if (header.type == RTM_IFINFO2 && header.length >= sizeof(struct if_msghdr2)) {
      struct if_msghdr2 message;
      memcpy(&message, bytes + offset, sizeof(message));
      // Ethernet/Wi-Fi links only: do not double-count loopback and VPN tunnels.
      if ((message.ifm_flags & IFF_UP) && !(message.ifm_flags & IFF_LOOPBACK) && message.ifm_data.ifi_type == IFT_ETHER) {
        result[[NSString stringWithFormat:@"%u", message.ifm_index]] = @[@(message.ifm_data.ifi_ibytes), @(message.ifm_data.ifi_obytes)];
      }
    }
    offset += header.length;
  }
  if (offset != size) return nil;
  return result.count ? result : nil;
}
static NSArray *Rates(NSDictionary *before, NSDictionary *after, double seconds) {
  if (!before.count || !after.count || !isfinite(seconds) || seconds <= 0 || ![[NSSet setWithArray:before.allKeys] isEqualToSet:[NSSet setWithArray:after.allKeys]]) return nil;
  long double first = 0, second = 0;
  for (NSString *key in before) {
    NSArray *a = before[key], *b = after[key];
    if (a.count != 2 || b.count != 2) return nil;
    uint64_t a0 = [a[0] unsignedLongLongValue], a1 = [a[1] unsignedLongLongValue];
    uint64_t b0 = [b[0] unsignedLongLongValue], b1 = [b[1] unsignedLongLongValue];
    if (b0 < a0 || b1 < a1) return nil;
    first += b0 - a0;
    second += b1 - a1;
  }
  double x = (double)(first / seconds), y = (double)(second / seconds);
  return isfinite(x) && isfinite(y) ? @[@(x), @(y)] : nil;
}
static NSString *Thermal(void) {
  switch ([NSProcessInfo processInfo].thermalState) {
    case NSProcessInfoThermalStateNominal: return @"Nominal";
    case NSProcessInfoThermalStateFair: return @"Fair";
    case NSProcessInfoThermalStateSerious: return @"Serious";
    case NSProcessInfoThermalStateCritical: return @"Critical";
  }
  return nil;
}

static NSDictionary *Collect(unsigned interval, NSError **error) {
  ReportAPI api = {0};
  void *subscription = NULL;
  CFMutableDictionaryRef subscribed = NULL;
  CFDictionaryRef first = NULL, second = NULL, delta = NULL;
  @try {
    if (!LoadAPI(&api, error)) return nil;
    NSDictionary *requested = RequestedChannels(&api, error);
    if (!requested) return nil;
    subscription = api.subscribe(NULL, (__bridge CFMutableDictionaryRef)requested, &subscribed, 0, NULL);
    if (!subscription || !subscribed) { Fail(error, @"IOReport could not subscribe to supported counters as the current user."); return nil; }
    NSDictionary *tables = FrequencyTables();
    NSMutableArray<NSString *> *warnings = [NSMutableArray array];
    first = api.sample(subscription, subscribed, NULL);
    double start = Monotonic();
    if (!first || !isfinite(start)) { Fail(error, @"IOReport could not collect a baseline sample."); return nil; }
    NSDictionary *diskBefore = DiskCounters();
    double diskStart = Monotonic();
    NSDictionary *networkBefore = NetworkCounters();
    double networkStart = Monotonic();
    double remaining = start + interval / 1000.0 - Monotonic();
    if (remaining > 0 && isfinite(remaining)) {
      struct timespec wait = {(time_t)remaining, (long)((remaining - floor(remaining)) * 1e9)};
      while (nanosleep(&wait, &wait) != 0) if (errno != EINTR) { Fail(error, @"The sample interval could not be measured."); return nil; }
    }
    second = api.sample(subscription, subscribed, NULL);
    double stop = Monotonic();
    if (!second || !isfinite(stop) || stop <= start) { Fail(error, @"IOReport could not collect a valid second sample."); return nil; }
    delta = api.delta(first, second, NULL);
    if (!delta) { Fail(error, @"IOReport could not calculate the sample delta."); return nil; }
    double seconds = stop - start;
    NSDictionary *diskAfter = DiskCounters();
    double diskStop = Monotonic();
    NSDictionary *networkAfter = NetworkCounters();
    double networkStop = Monotonic();
    NSArray *diskRates = Rates(diskBefore, diskAfter, diskStop - diskStart);
    NSArray *networkRates = Rates(networkBefore, networkAfter, networkStop - networkStart);
    NSArray *samples = Array(((__bridge NSDictionary *)delta)[@"IOReportChannels"]);
    if (!samples || samples.count > 4096) { Fail(error, @"Invalid or oversized IOReport sample."); return nil; }
    NSMutableDictionary *power = [NSMutableDictionary dictionary];
    NSMutableSet *invalidPower = [NSMutableSet set];
    NSMutableSet *seen = [NSMutableSet set];
    NSMutableDictionary<NSString *, NSMutableArray *> *cores = [NSMutableDictionary dictionary];
    NSDictionary *gpu = nil;
    for (id value in samples) {
      NSDictionary *sample = Dict(value);
      if (!sample) continue;
      CFDictionaryRef item = (__bridge CFDictionaryRef)sample;
      NSString *group = CFText(api.group(item)), *subgroup = CFText(api.subgroup(item));
      NSString *channel = CFText(api.channel(item));
      NSString *unit = [CFText(api.unit(item)) stringByTrimmingCharactersInSet:NSCharacterSet.whitespaceAndNewlineCharacterSet];
      NSString *identity = [NSString stringWithFormat:@"%llu:%llu", (unsigned long long)api.driverID(item), (unsigned long long)api.channelID(item)];
      if ([seen containsObject:identity]) continue;
      [seen addObject:identity];
      if ([group isEqualToString:@"CPU Stats"] && [subgroup isEqualToString:@"CPU Core Performance States"]) {
        NSString *tier = [channel containsString:@"ECPU"] ? @"ECPU" : [channel containsString:@"PCPU"] ? @"PCPU" : nil;
        NSDictionary *reading = tier ? StateMetrics(ReadStates(&api, item), tables[tier], NO) : nil;
        if (reading) {
          if (!cores[tier]) cores[tier] = [NSMutableArray array];
          [cores[tier] addObject:reading];
        }
        continue;
      }
      if ([group isEqualToString:@"GPU Stats"] && [subgroup isEqualToString:@"GPU Performance States"] && [channel isEqualToString:@"GPUPH"]) {
        gpu = StateMetrics(ReadStates(&api, item), tables[@"GPU"], YES);
        continue;
      }
      NSString *component = Component(channel);
      BOOL energy = Wanted(group, subgroup, channel, unit);
      if (!component || !energy) continue;
      NSString *key = [NSString stringWithFormat:@"%@:%@", component, group];
      NSNumber *mw = Milliwatts(api.integer(item, 0), unit, seconds);
      if (!mw) { [invalidPower addObject:key]; continue; }
      double sum = [power[key] doubleValue] + mw.doubleValue;
      if (!isfinite(sum)) { [invalidPower addObject:key]; continue; }
      power[key] = @(sum);
    }
    NSMutableDictionary *values = [NSMutableDictionary dictionary];
    NSMutableDictionary *sources = [NSMutableDictionary dictionary];
    for (NSString *component in @[@"cpu", @"gpu", @"ane"]) {
      values[component] = Null();
      sources[component] = Null();
      for (NSString *source in @[@"CLPC", @"Energy Model", @"PMP"]) {
        NSString *key = [NSString stringWithFormat:@"%@:%@", component, source];
        if (power[key] && ![invalidPower containsObject:key]) { values[component] = power[key]; sources[component] = source; break; }
      }
      if (values[component] == Null()) Warn(warnings, [NSString stringWithFormat:@"%@ energy counter unavailable or invalid.", component.uppercaseString]);
      if ([NSProcessInfo processInfo].operatingSystemVersion.majorVersion == 27
          && [sources[component] isEqual:@"Energy Model"] && ![component isEqualToString:@"gpu"]) {
        Warn(warnings, [NSString stringWithFormat:@"%@ uses the legacy Energy Model fallback; this counter can stall on macOS 27 when CLPC is unavailable.", component.uppercaseString]);
      }
    }
    NSMutableArray *clusters = [NSMutableArray array];
    for (NSString *tier in @[@"ECPU", @"PCPU"]) {
      NSArray *readings = cores[tier];
      if (!readings.count) continue;
      double total = 0, active = 0, weighted = 0;
      BOOL frequenciesValid = YES;
      for (NSDictionary *reading in readings) {
        total += [reading[@"total"] doubleValue];
        active += [reading[@"active"] doubleValue];
        NSNumber *sum = Number(reading[@"weighted"]);
        if (!sum) frequenciesValid = NO; else weighted += sum.doubleValue;
      }
      id frequency = frequenciesValid && active > 0 && isfinite(weighted) ? @(weighted / active) : Null();
      [clusters addObject:@{@"name":tier, @"active_percent":total > 0 ? @(100 * active / total) : Null(), @"frequency_mhz":frequency}];
      if (!frequenciesValid) Warn(warnings, [NSString stringWithFormat:@"%@ frequency table does not match the reported states.", tier]);
    }
    if (!clusters.count) Warn(warnings, @"CPU performance-state counters are unavailable.");
    if (!gpu) Warn(warnings, @"GPU performance-state counters are unavailable.");
    else if (!Number(gpu[@"weighted"])) Warn(warnings, @"GPU frequency table does not match the reported states.");
    if (!diskRates) Warn(warnings, @"Disk byte-rate counters are unavailable or changed during this interval.");
    if (!networkRates) Warn(warnings, @"Ethernet/Wi-Fi byte-rate counters are unavailable or changed during this interval.");
    return @{
      @"schema_version":@1, @"source":@"ioreport", @"effective_uid":@(geteuid()),
      @"sampled_at_unix_ms":@((uint64_t)(NSDate.date.timeIntervalSince1970 * 1000)), @"sample_duration_ms":@(seconds * 1000),
      @"cpu_power_mw":values[@"cpu"], @"gpu_power_mw":values[@"gpu"], @"ane_power_mw":values[@"ane"],
      @"gpu_active_percent":gpu[@"active_percent"] ?: Null(), @"gpu_frequency_mhz":gpu[@"frequency_mhz"] ?: Null(),
      @"cpu_clusters":clusters, @"thermal_pressure":Thermal() ?: Null(),
      @"disk_read_bytes_per_second":diskRates ? diskRates[0] : Null(), @"disk_write_bytes_per_second":diskRates ? diskRates[1] : Null(),
      @"network_in_bytes_per_second":networkRates ? networkRates[0] : Null(), @"network_out_bytes_per_second":networkRates ? networkRates[1] : Null(),
      @"power_sources":sources, @"warnings":warnings,
    };
  } @finally {
    if (delta) CFRelease(delta);
    if (second) CFRelease(second);
    if (first) CFRelease(first);
    if (subscribed) CFRelease(subscribed);
    if (subscription) CFRelease((CFTypeRef)subscription);
    if (api.library) dlclose(api.library);
  }
}

static BOOL SelfTest(void) {
  if (fabs(Milliwatts(1000000000, @"nJ", 0.5).doubleValue - 2000) > 1e-9) return NO;
  if (fabs(Milliwatts(500000, @"uJ", 2).doubleValue - 250) > 1e-9) return NO;
  if (fabs(Milliwatts(500, @"mJ", 2).doubleValue - 250) > 1e-9) return NO;
  if (!Milliwatts(0, @"nJ", 0.5) || Milliwatts(0, @"nJ", 0.5).doubleValue != 0) return NO;
  if (Milliwatts(-1, @"nJ", 1) || Milliwatts(1, @"unknown", 1) || Milliwatts(1, @"nJ", 0)) return NO;
  NSArray *states = @[@{@"name":@"DOWN",@"residency":@800}, @{@"name":@"IDLE",@"residency":@100},
    @{@"name":@"V0",@"residency":@25}, @{@"name":@"V1",@"residency":@75}];
  NSDictionary *reading = StateMetrics(states, @[@1000,@2000], NO);
  if (fabs([reading[@"active_percent"] doubleValue] - 10) > 1e-9 || fabs([reading[@"frequency_mhz"] doubleValue] - 1750) > 1e-9) return NO;
  if (Number(StateMetrics(states, @[@1000], NO)[@"frequency_mhz"])) return NO;
  if (StateMetrics(@[@{@"name":@"IDLE",@"residency":@-1}], @[@1000], NO)) return NO;
  NSDictionary *before = @{@"disk":@[@100,@200]}, *after = @{@"disk":@[@200,@500]};
  NSArray *rates = Rates(before, after, 2);
  if ([rates[0] doubleValue] != 50 || [rates[1] doubleValue] != 150 || Rates(after, before, 2) || Rates(before, @{}, 2)) return NO;
  return YES;
}

int main(int argc, const char *argv[]) {
  @autoreleasepool {
    @try {
      if (argc == 2 && strcmp(argv[1], "--self-test") == 0) {
        if (!SelfTest()) { fprintf(stderr, "Native telemetry self-test failed.\n"); return 1; }
        puts("{\"self_test\":true}");
        return 0;
      }
      unsigned interval = 500;
      if (argc != 1) {
        if (argc != 3 || strcmp(argv[1], "--interval-ms") != 0) { fprintf(stderr, "Usage: alloy-metrics [--interval-ms 100..2000] | --self-test\n"); return 2; }
        char *end = NULL;
        errno = 0;
        unsigned long parsed = strtoul(argv[2], &end, 10);
        if (errno || !argv[2][0] || !end || *end || parsed < 100 || parsed > 2000) { fprintf(stderr, "Sample interval must be an integer from 100 to 2000 milliseconds.\n"); return 2; }
        interval = (unsigned)parsed;
      }
      NSError *error = nil;
      NSDictionary *sample = Collect(interval, &error);
      if (!sample) { fprintf(stderr, "alloy-metrics: %s\n", (error.localizedDescription ?: @"Native sampling failed.").UTF8String); return 1; }
      NSData *json = [NSJSONSerialization dataWithJSONObject:sample options:0 error:&error];
      if (!json || json.length > 1024 * 1024) { fprintf(stderr, "alloy-metrics: Invalid or oversized telemetry JSON.\n"); return 1; }
      if (fwrite(json.bytes, 1, json.length, stdout) != json.length || fputc('\n', stdout) == EOF || fflush(stdout) != 0) return 1;
      return 0;
    } @catch (NSException *exception) {
      fprintf(stderr, "alloy-metrics: %s\n", exception.reason.UTF8String ?: "Native sampling exception.");
      return 1;
    }
  }
}
