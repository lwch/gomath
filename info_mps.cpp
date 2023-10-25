#include "info.h"

#if defined(__APPLE__) && defined(__arm64__)

bool has_avx() { return false; }
bool has_avx512() { return false; }

#endif