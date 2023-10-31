#include "../../../../info.h"
#include "../exception.hpp"
#include "../tensor.h"
#include "common.hpp"
#include "fp32.h"
#include <exception>
#include <stdio.h>

#if !(defined(__APPLE__) && defined(__arm64__))

#define COMPUTE_VECTOR(fn, op)                                                 \
  do {                                                                         \
    size_t np = (d & ~(bs - 1));                                               \
    for (size_t i = 0; i < np; i += bs) {                                      \
      T x_vec = _fp32_loadu<T>(x);                                             \
      T w_vec = _fp32_loadu<T>(w);                                             \
      T y_vec = fn<T>(x_vec, w_vec);                                           \
      _fp32_store_ps(y, y_vec);                                                \
      x += bs;                                                                 \
      w += bs;                                                                 \
      y += bs;                                                                 \
    }                                                                          \
    for (size_t i = 0; i < d - np; i++) {                                      \
      y[i] = x[i] op w[i];                                                     \
    }                                                                          \
  } while (0)

#define COMPUTE_SCALAR(fn, op)                                                 \
  do {                                                                         \
    size_t np = (d & ~(bs - 1));                                               \
    T x_vec = _fp32_set1_ps<T>(x);                                             \
    for (size_t i = 0; i < np; i += bs) {                                      \
      T w_vec = _fp32_loadu<T>(w);                                             \
      T y_vec = fn<T>(x_vec, w_vec);                                           \
      _fp32_store_ps(y, y_vec);                                                \
      w += bs;                                                                 \
      y += bs;                                                                 \
    }                                                                          \
    for (size_t i = 0; i < d - np; i++) {                                      \
      y[i] = x op w[i];                                                        \
    }                                                                          \
  } while (0)

template <typename T> class fp32_impl : public fp32 {
public:
  // dot_vector
  virtual float dot_vector(const float *x, const float *w, int64_t d) {
    size_t np = (d & ~(bs - 1));
    T y_vec = _fp32_zero_ps<T>();
    for (size_t i = 0; i < np; i += bs) {
      T x_vec = _fp32_loadu<T>(x);
      T w_vec = _fp32_loadu<T>(w);
      y_vec = _fp32_fmadd_ps<T>(x_vec, w_vec, y_vec);
      x += bs;
      w += bs;
    }
    float y = _fp32_sum_ps<T>(y_vec);
    for (size_t i = 0; i < d - np; i++) {
      y += x[i] * w[i];
    }
    return y;
  }

  virtual void mul_scalar(const float x, const float *w, float *y, int64_t d) {
    COMPUTE_SCALAR(_fp32_mul_ps, *);
  }

  virtual void mul_vector(const float *x, const float *w, float *y, int64_t d) {
    COMPUTE_VECTOR(_fp32_mul_ps, *);
  }

  virtual void div_scalar(const float x, const float *w, float *y, int64_t d) {
    COMPUTE_SCALAR(_fp32_div_ps, /);
  }

  virtual void div_vector(const float *x, const float *w, float *y, int64_t d) {
    COMPUTE_VECTOR(_fp32_div_ps, /);
  }
};

static fp32 *_fp32;

bool fp32_init() {
  if (has_avx512()) {
    _fp32 = new fp32_impl<__m512>();
    return true;
  } else if (has_avx()) {
    _fp32 = new fp32_impl<__m256>();
    return true;
  }
  return false;
}

float fp32_dot_vector(const float *x, const float *w, int64_t d) {
  return _fp32->dot_vector(x, w, d);
}

void fp32_mul_vector(const float *x, const float *w, float *y, int64_t d) {
  _fp32->mul_vector(x, w, y, d);
}

void fp32_mul_scalar(const float x, const float *w, float *y, int64_t d) {
  _fp32->mul_scalar(x, w, y, d);
}

void fp32_div_scalar(const float x, const float *w, float *y, int64_t d) {
  _fp32->div_scalar(x, w, y, d);
}

void fp32_div_vector(const float *x, const float *w, float *y, int64_t d) {
  _fp32->div_vector(x, w, y, d);
}

#endif