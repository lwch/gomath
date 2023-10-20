#include "info.h"

bool has_avx() {
#ifdef __AVX__
  return true;
#else
  return false;
#endif
}

bool has_avx2() {
#ifdef __AVX2__
  return true;
#else
  return false;
#endif
}

bool has_avx512() {
#ifdef __AVX512F__
  return true;
#else
  return false;
#endif
}

bool has_sse3() {
#if defined(__SSE3__)
  return true;
#else
  return false;
#endif
}

bool has_ssse3() {
#if defined(__SSSE3__)
  return true;
#else
  return false;
#endif
}

bool has_neon() {
#if defined(__ARM_NEON)
  return true;
#else
  return false;
#endif
}