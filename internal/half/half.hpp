#ifndef __INTERNAL_HALF_HALF_H__
#define __INTERNAL_HALF_HALF_H__

#ifdef __cplusplus
extern "C" {
#endif

#include "_cgo_export.h"
#include <stdint.h>

inline float fp16_to_fp32(uint16_t n) {
  extern GoFloat32 Decode(GoUint16);
  return Decode(n);
}

inline uint16_t fp32_to_fp16(float n) {
  extern GoUint16 Encode(GoFloat32);
  return Encode(n);
}

#ifdef __cplusplus
}
#endif

#endif