#include "fp16.h"

#if defined(__APPLE__) && defined(__arm64__)

bool fp16_init() { return false; }
uint16_t fp16_dot(const uint16_t *, const uint16_t *, int64_t) { return 0; }
void fp16_scalar_mul(const uint16_t x, const uint16_t *w, uint16_t *y,
                     int64_t d) {}
void fp16_mul(const uint16_t *, const uint16_t *, uint16_t *, int64_t) {}
void fp16_scalar_div(const uint16_t x, const uint16_t *w, uint16_t *y,
                     int64_t d) {}
void fp16_div(const uint16_t *, const uint16_t *, uint16_t *, int64_t) {}
void fp16_scalar_add(const uint16_t x, const uint16_t *w, uint16_t *y,
                     int64_t d) {}
void fp16_add(const uint16_t *, const uint16_t *, uint16_t *, int64_t) {}
void fp16_scalar_sub(const uint16_t x, const uint16_t *w, uint16_t *y,
                     int64_t d) {}
void fp16_sub(const uint16_t *, const uint16_t *, uint16_t *, int64_t) {}
void fp16_pow(const uint16_t *, const uint16_t, uint16_t *, int64_t) {}

#endif