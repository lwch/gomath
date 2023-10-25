#include "../info.h"
#include "../internal/half/half.hpp"
#include "exception.hpp"
#include "f16.h"
#include <stdio.h>

#if !(defined(__APPLE__) && defined(__arm64__))

#include <immintrin.h>

typedef uint16_t (*fn_f16_dot_vector)(uint16_t *, uint16_t *, int64_t);
static fn_f16_dot_vector f16_dot_vector_impl = nullptr;

bool f16_init() {
  if (has_avx512()) {
    extern uint16_t f16_dot_vector_avx512(uint16_t *, uint16_t *, int64_t);
    f16_dot_vector_impl = f16_dot_vector_avx512;
    return true;
  } else if (has_avx()) {
    extern uint16_t f16_dot_vector_avx(uint16_t *, uint16_t *, int64_t);
    f16_dot_vector_impl = f16_dot_vector_avx;
    return true;
  }
  return false;
}

uint16_t f16_dot_vector(uint16_t *x, uint16_t *w, int64_t d) {
  return f16_dot_vector_impl(x, w, d);
}

inline __m256 _mm256_f16_load(uint16_t *x) {
#ifdef __F16C__
  __m128i x_vec = _mm_loadu_si128((__m128i *)x);
  return _mm256_cvtph_ps(x_vec);
#else
  float tmp[8];
  for (size_t i = 0; i < 8; i++) {
    tmp[i] = f16_to_f32(x[i]);
  }
  return _mm256_loadu_ps(tmp);
#endif
}

inline __m512 _mm512_f16_load(uint16_t *x) {
#ifdef __F16C__
  __m256i x_vec = _mm256_loadu_si256((__m256i *)x);
  return _mm512_cvtph_ps(x_vec);
#else
  float tmp[16];
  for (size_t i = 0; i < 16; i++) {
    tmp[i] = f16_to_f32(x[i]);
  }
  return _mm512_loadu_ps(tmp);
#endif
}

uint16_t f16_dot_vector_avx(uint16_t *x, uint16_t *w, int64_t d) {
  const size_t bs = 8;
  size_t np = (d & ~(bs - 1));
  __m256 y_vec = _mm256_setzero_ps();
  for (size_t i = 0; i < np; i += bs) {
    __m256 x_vec = _mm256_f16_load(x);
    __m256 w_vec = _mm256_f16_load(w);
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
  for (size_t i = 0; i < d - np; i++) {
    y += f16_to_f32(x[i]) * f16_to_f32(w[i]);
  }
  return f32_to_f16(y);
}

uint16_t f16_dot_vector_avx512(uint16_t *x, uint16_t *w, int64_t d) {
  const size_t bs = 16;
  size_t np = (d & ~(bs - 1));
  __m512 y_vec = _mm512_setzero_ps();
  for (size_t i = 0; i < np; i += bs) {
    __m512 x_vec = _mm512_f16_load(x);
    __m512 w_vec = _mm512_f16_load(w);
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
  for (size_t i = 0; i < d - np; i++) {
    y += f16_to_f32(x[i]) * f16_to_f32(w[i]);
  }
  return f32_to_f16(y);
}

#endif