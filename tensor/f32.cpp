#include "f32.h"
#include "../info.h"
#include "exception.hpp"
#include <stdio.h>

#if defined(__APPLE__) && defined(__arm64__)

void f32_init() {}
float f32_dot_vector(float *, float *, int64_t) { return 0; }

#else

#include <immintrin.h>

typedef float (*fn_f32_dot_vector)(float *, float *, int64_t);
fn_f32_dot_vector f_f32_dot_vector = nullptr;

void f32_init() {
  if (has_avx512()) {
    extern float f32_dot_vector_avx512(float *, float *, int64_t);
    f_f32_dot_vector = f32_dot_vector_avx512;
  } else if (has_avx()) {
    extern float f32_dot_vector_avx(float *, float *, int64_t);
    f_f32_dot_vector = f32_dot_vector_avx;
  }
}

float f32_dot_vector(float *x, float *w, int64_t d) {
  return f_f32_dot_vector(x, w, d);
}

float f32_dot_vector_avx(float *x, float *w, int64_t d) {
  const size_t bs = 8;
  size_t np = (d & ~(bs - 1));
  __m256 y_vec = _mm256_setzero_ps();
  for (size_t i = 0; i < np; i += bs) {
    __m256 x_vec = _mm256_loadu_ps(x);
    __m256 w_vec = _mm256_loadu_ps(w);
#ifdef __FMA__
    y_vec = _mm256_fmadd_ps(x_vec, w_vec, y_vec);
#else
    y_vec = _mm256_add_ps(_mm256_mul_ps(x_vec, w_vec), y_vec);
#endif
    x += bs;
    w += bs;
  }
  __m128 t0 = _mm_add_ps(_mm256_castps256_ps128(y_vec),
                         _mm256_extractf128_ps(y_vec, 1));
  __m128 t1 = _mm_hadd_ps(t0, t0);
  t1 = _mm_hadd_ps(t1, t1);
  float y = _mm_cvtss_f32(t1);
  for (size_t i = np; i < d; i++) {
    y += x[i] * w[i];
  }
  return y;
}

float f32_dot_vector_avx512(float *x, float *w, int64_t d) {
  const size_t bs = 16;
  size_t np = (d & ~(bs - 1));
  __m512 y_vec = _mm512_setzero_ps();
  for (size_t i = 0; i < np; i += bs) {
    __m512 x_vec = _mm512_loadu_ps(x);
    __m512 w_vec = _mm512_loadu_ps(w);
#ifdef __FMA__
    y_vec = _mm512_fmadd_ps(x_vec, w_vec, y_vec);
#else
    y_vec = _mm512_add_ps(_mm512_mul_ps(x_vec, w_vec), y_vec);
#endif
    x += bs;
    w += bs;
  }
  __m256 t0 = _mm256_add_ps(_mm512_castps512_ps256(y_vec),
                            _mm512_extractf32x8_ps(y_vec, 1));
  __m128 t1 =
      _mm_add_ps(_mm256_castps256_ps128(t0), _mm256_extractf128_ps(t0, 1));
  __m128 t2 = _mm_hadd_ps(t1, t1);
  t2 = _mm_hadd_ps(t2, t2);
  float y = _mm_cvtss_f32(t2);
  for (size_t i = np; i < d; i++) {
    y += x[i] * w[i];
  }
  return y;
}

#endif