#ifndef __INTERNAL_HALF_HALF_H__
#define __INTERNAL_HALF_HALF_H__

#ifdef __cplusplus
extern "C" {
#endif

#include "_cgo_export.h"
#include <stdint.h>

inline float f16_to_f32(uint16_t n) {
  extern GoFloat32 Decode(GoUint16);
  return Decode(n);
}

inline uint16_t f32_to_f16(float n) {
  extern GoUint16 Encode(GoFloat32);
  return Encode(n);
}

#ifdef __cplusplus
}
#endif

#endif