#include "f16.h"

#if defined(__APPLE__) && defined(__arm64__)

bool f16_init() { return false; }
uint16_t f16_dot_vector(uint16_t *, uint16_t *, int64_t) { return 0; }

#endif