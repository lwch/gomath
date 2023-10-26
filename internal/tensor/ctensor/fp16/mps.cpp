#include "fp16.h"

#if defined(__APPLE__) && defined(__arm64__)

bool fp16_init() { return false; }
uint16_t fp16_dot_vector(const uint16_t *, const uint16_t *, int64_t) {
  return 0;
}
void fp16_mul_vector(const uint16_t *, const uint16_t *, uint16_t *, int64_t) {}
void fp16_mul_scalar(const uint16_t *x, uint16_t w, uint16_t *y, int64_t d) {}
void fp16_div_vector(const uint16_t *, const uint16_t *, uint16_t *, int64_t) {}

#endif