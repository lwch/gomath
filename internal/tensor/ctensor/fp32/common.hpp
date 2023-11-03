#ifndef __TENSOR_FP32_COMMON_HPP__
#define __TENSOR_FP32_COMMON_HPP__

#if !(defined(__APPLE__) && defined(__arm64__))

#include <immintrin.h>

class fp32 {
public:
  fp32() {}
  virtual ~fp32() {}
  virtual float dot(const float *x, const float *w, int64_t d) = 0;
  virtual void mul(const float x, const float *w, float *y, int64_t d) = 0;
  virtual void mul(const float *x, const float *w, float *y, int64_t d) = 0;
  virtual void div(const float x, const float *w, float *y, int64_t d) = 0;
  virtual void div(const float *x, const float *w, float *y, int64_t d) = 0;
  virtual void add(const float x, const float *w, float *y, int64_t d) = 0;
  virtual void add(const float *x, const float *w, float *y, int64_t d) = 0;
  virtual void sub(const float x, const float *w, float *y, int64_t d) = 0;
  virtual void sub(const float *x, const float *w, float *y, int64_t d) = 0;
  virtual void pow(const float *x, const float n, float *y, int64_t d) = 0;
};

// load fp32 into register

template <typename T> inline T _fp32_loadu(const float *x) {
  throw not_implemented_exception("_fp32_loadu");
}

template <> inline __m256 _fp32_loadu(const float *x) {
  return _mm256_loadu_ps(x);
}

template <> inline __m512 _fp32_loadu(const float *x) {
  return _mm512_loadu_ps(x);
}

// get zero register

template <typename T> inline T _fp32_zero_ps() {
  throw not_implemented_exception("_fp32_zero");
}

template <> inline __m256 _fp32_zero_ps() { return _mm256_setzero_ps(); }
template <> inline __m512 _fp32_zero_ps() { return _mm512_setzero_ps(); }

// fmadd

template <typename T>
inline T _fp32_fmadd_ps(const T &x, const T &y, const T &z) {
  throw not_implemented_exception("_fp32_fmadd_ps");
}

template <>
inline __m256 _fp32_fmadd_ps(const __m256 &x, const __m256 &y,
                             const __m256 &z) {
  return _mm256_fmadd_ps(x, y, z);
}

template <>
inline __m512 _fp32_fmadd_ps(const __m512 &x, const __m512 &y,
                             const __m512 &z) {
  return _mm512_fmadd_ps(x, y, z);
}

// sum reduce

template <typename T> inline float _fp32_sum_ps(const T &vec) {
  throw not_implemented_exception("_fp32_sum_ps");
}

template <> inline float _fp32_sum_ps(const __m256 &vec) {
  __m128 t0 =
      _mm_add_ps(_mm256_castps256_ps128(vec), _mm256_extractf128_ps(vec, 1));
  __m128 t1 = _mm_hadd_ps(t0, t0);
  t1 = _mm_hadd_ps(t1, t1);
  return _mm_cvtss_f32(t1);
}

template <> inline float _fp32_sum_ps(const __m512 &vec) {
  __m256 t0 = _mm256_add_ps(_mm512_castps512_ps256(vec),
                            _mm512_extractf32x8_ps(vec, 1));
  __m128 t1 =
      _mm_add_ps(_mm256_castps256_ps128(t0), _mm256_extractf128_ps(t0, 1));
  __m128 t2 = _mm_hadd_ps(t1, t1);
  t2 = _mm_hadd_ps(t2, t2);
  return _mm_cvtss_f32(t2);
}

// mul

template <typename T> inline T _fp32_mul_ps(const T &x, const T &y) {
  throw not_implemented_exception("_fp32_mul_ps");
}

template <> inline __m256 _fp32_mul_ps(const __m256 &x, const __m256 &y) {
  return _mm256_mul_ps(x, y);
}

template <> inline __m512 _fp32_mul_ps(const __m512 &x, const __m512 &y) {
  return _mm512_mul_ps(x, y);
}

// div

template <typename T> inline T _fp32_div_ps(const T &x, const T &y) {
  throw not_implemented_exception("_fp32_div_ps");
}

template <> inline __m256 _fp32_div_ps(const __m256 &x, const __m256 &y) {
  return _mm256_div_ps(x, y);
}

template <> inline __m512 _fp32_div_ps(const __m512 &x, const __m512 &y) {
  return _mm512_div_ps(x, y);
}

// add

template <typename T> inline T _fp32_add_ps(const T &x, const T &y) {
  throw not_implemented_exception("_fp32_add_ps");
}

template <> inline __m256 _fp32_add_ps(const __m256 &x, const __m256 &y) {
  return _mm256_add_ps(x, y);
}

template <> inline __m512 _fp32_add_ps(const __m512 &x, const __m512 &y) {
  return _mm512_add_ps(x, y);
}

// sub

template <typename T> inline T _fp32_sub_ps(const T &x, const T &y) {
  throw not_implemented_exception("_fp32_sub_ps");
}

template <> inline __m256 _fp32_sub_ps(const __m256 &x, const __m256 &y) {
  return _mm256_sub_ps(x, y);
}

template <> inline __m512 _fp32_sub_ps(const __m512 &x, const __m512 &y) {
  return _mm512_sub_ps(x, y);
}

// store

template <typename T> void _fp32_store_ps(float *y, const T &x) {
  throw not_implemented_exception("_fp32_store_ps");
}

template <> inline void _fp32_store_ps(float *y, const __m256 &x) {
  _mm256_storeu_ps(y, x);
}

template <> inline void _fp32_store_ps(float *y, const __m512 &x) {
  _mm512_storeu_ps(y, x);
}

// set1

template <typename T> inline T _fp32_set1_ps(float x) {
  throw not_implemented_exception("_fp32_set1_ps");
}

template <> inline __m256 _fp32_set1_ps(float x) { return _mm256_set1_ps(x); }

template <> inline __m512 _fp32_set1_ps(float x) { return _mm512_set1_ps(x); }

#endif

#endif