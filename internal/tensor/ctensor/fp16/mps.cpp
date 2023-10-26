#include "fp16.h"

#if defined(__APPLE__) && defined(__arm64__)

bool fp16_init() { return false; }
uint16_t fp16_dot_vector(const uint16_t *, const uint16_t *, int64_t) {
  return 0;
}
void fp16_mul_vector(const uint16_t *, const uint16_t *, uint16_t *, int64_t) {}

#endif