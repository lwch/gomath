#include "fp32.h"

#if defined(__APPLE__) && defined(__arm64__)

bool fp32_init() { return false; }
float fp32_dot_vector(float *, float *, int64_t) { return 0; }

#endif