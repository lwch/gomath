#ifndef __CTENSOR_FP16_COMMON_HPP__
#define __CTENSOR_FP16_COMMON_HPP__

#if !(defined(__APPLE__) && defined(__arm64__))

#include <immintrin.h>
#include <stdint.h>

class fp16 {
public:
  fp16() {}
  virtual ~fp16() {}
  virtual uint16_t dot_vector(const uint16_t *x, const uint16_t *w,
                              int64_t d) = 0;
  virtual void mul_vector(const uint16_t *x, const uint16_t *w, uint16_t *y,
                          int64_t d) = 0;
  virtual void mul_scalar(const uint16_t *x, const uint16_t w, uint16_t *y,
                          int64_t d) = 0;
  virtual void div_vector(const uint16_t *x, const uint16_t *w, uint16_t *y,
                          int64_t d) = 0;
};

// load fp16 into register

template <typename T> inline T _fp16_loadu(const uint16_t *x) {
  throw not_implemented_exception("_fp16_loadu");
}

template <> inline __m256 _fp16_loadu(const uint16_t *x) {
  __m128i x_vec = _mm_loadu_si128((__m128i *)x);
  return _mm256_cvtph_ps(x_vec);
}

template <> inline __m512 _fp16_loadu(const uint16_t *x) {
  __m256i x_vec = _mm256_loadu_si256((__m256i *)x);
  return _mm512_cvtph_ps(x_vec);
}

// get zero register

template <typename T> inline T _fp16_zero_ps() {
  throw not_implemented_exception("_fp16_zero");
}

template <> inline __m256 _fp16_zero_ps() { return _mm256_setzero_ps(); }
template <> inline __m512 _fp16_zero_ps() { return _mm512_setzero_ps(); }

// fmadd

template <typename T>
inline T _fp16_fmadd_ps(const T &x, const T &y, const T &z) {
  throw not_implemented_exception("_fp16_fmadd_ps");
}

template <>
inline __m256 _fp16_fmadd_ps(const __m256 &x, const __m256 &y,
                             const __m256 &z) {
  return _mm256_fmadd_ps(x, y, z);
}

template <>
inline __m512 _fp16_fmadd_ps(const __m512 &x, const __m512 &y,
                             const __m512 &z) {
  return _mm512_fmadd_ps(x, y, z);
}

// sum reduce

template <typename T> inline float _fp16_sum_ps(const T &vec) {
  throw not_implemented_exception("_fp16_sum_ps");
}

template <> inline float _fp16_sum_ps(const __m256 &vec) {
  __m128 t0 =
      _mm_add_ps(_mm256_castps256_ps128(vec), _mm256_extractf128_ps(vec, 1));
  __m128 t1 = _mm_hadd_ps(t0, t0);
  t1 = _mm_hadd_ps(t1, t1);
  return _mm_cvtss_f32(t1);
}

template <> inline float _fp16_sum_ps(const __m512 &vec) {
  __m256 t0 = _mm256_add_ps(_mm512_castps512_ps256(vec),
                            _mm512_extractf32x8_ps(vec, 1));
  __m128 t1 =
      _mm_add_ps(_mm256_castps256_ps128(t0), _mm256_extractf128_ps(t0, 1));
  __m128 t2 = _mm_hadd_ps(t1, t1);
  t2 = _mm_hadd_ps(t2, t2);
  return _mm_cvtss_f32(t2);
}

// mul

template <typename T, typename T2>
inline T2 _fp16_mul_ps(const T &x, const T &y) {
  throw not_implemented_exception("_fp16_mul_ps");
}

template <> inline __m128i _fp16_mul_ps(const __m256 &x, const __m256 &y) {
  return _mm256_cvtps_ph(_mm256_mul_ps(x, y), 0);
}

template <> inline __m256i _fp16_mul_ps(const __m512 &x, const __m512 &y) {
  return _mm512_cvtps_ph(_mm512_mul_ps(x, y), 0);
}

// div

template <typename T, typename T2>
inline T2 _fp16_div_ps(const T &x, const T &y) {
  throw not_implemented_exception("_fp16_div_ps");
}

template <> inline __m128i _fp16_div_ps(const __m256 &x, const __m256 &y) {
  return _mm256_cvtps_ph(_mm256_div_ps(x, y), 0);
}

template <> inline __m256i _fp16_div_ps(const __m512 &x, const __m512 &y) {
  return _mm512_cvtps_ph(_mm512_div_ps(x, y), 0);
}

// store

template <typename T> void _fp16_store_ps(uint16_t *y, const T &x) {
  throw not_implemented_exception("_fp16_store_ps");
}

template <> inline void _fp16_store_ps(uint16_t *y, const __m128i &x) {
  _mm_storeu_si128((__m128i *)y, x);
}

template <> inline void _fp16_store_ps(uint16_t *y, const __m256i &x) {
  _mm256_storeu_si256((__m256i *)y, x);
}

// set1

template <typename T> inline T _fp16_set1_ps(float x) {
  throw not_implemented_exception("_fp16_set1_ps");
}

template <> inline __m256 _fp16_set1_ps(float x) { return _mm256_set1_ps(x); }

template <> inline __m512 _fp16_set1_ps(float x) { return _mm512_set1_ps(x); }

#endif

#endif