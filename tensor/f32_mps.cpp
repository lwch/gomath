#include "f32.h"

#if defined(__APPLE__) && defined(__arm64__)

bool f32_init() { return false; }
float f32_dot_vector(float *, float *, int64_t) { return 0; }

#endif