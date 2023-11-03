#include "fp32.h"

#if defined(__APPLE__) && defined(__arm64__)

bool fp32_init() { return false; }
float fp32_dot(const float *, const float *, int64_t) { return 0; }
void fp32_scalar_mul(const float, const float *, float *, int64_t) {}
void fp32_mul(const float *, const float *, float *, int64_t) {}
void fp32_scalar_div(const float, const float *, float *, int64_t) {}
void fp32_div(const float *, const float *, float *, int64_t) {}
void fp32_scalar_add(const float, const float *, float *, int64_t) {}
void fp32_add(const float *, const float *, float *, int64_t) {}
void fp32_scalar_sub(const float, const float *, float *, int64_t) {}
void fp32_sub(const float *, const float *, float *, int64_t) {}
void fp32_pow(const float *, const float, float *, int64_t) {}

#endif