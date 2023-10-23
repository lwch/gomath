#include "bf16.h"
#include "../info.h"
#include "exception.hpp"
#include <stdio.h>

#if defined(__APPLE__) && defined(__arm64__)

void bf16_init() {}
float bf16_dot_vector(float *, float *, int64_t) { return 0; }

#else

#include <immintrin.h>
typedef float (*fn_bf16_dot_vector)(float *, float *, int64_t);
fn_bf16_dot_vector bf16_dot_vector_impl = nullptr;

void bf16_init() {
  if (has_avx512()) {
    extern float f32_dot_vector_avx512(float *, float *, int64_t);
    bf16_dot_vector_impl = f32_dot_vector_avx512;
  } else if (has_avx()) {
    extern float f32_dot_vector_avx(float *, float *, int64_t);
    bf16_dot_vector_impl = f32_dot_vector_avx;
  }
}

float bf16_dot_vector(float *x, float *w, int64_t d) {
  return bf16_dot_vector_impl(x, w, d);
}

#endif