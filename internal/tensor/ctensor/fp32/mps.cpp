#include "fp32.h"

#if defined(__APPLE__) && defined(__arm64__)

bool fp32_init() { return false; }
float fp32_dot_vector(const float *, const float *, int64_t) { return 0; }
void fp32_mul_scalar(const float, const float *, float *, int64_t) {}
void fp32_mul_vector(const float *, const float *, float *, int64_t) {}
void fp32_div_scalar(const float, const float *, float *, int64_t) {}
void fp32_div_vector(const float *, const float *, float *, int64_t) {}

#endif