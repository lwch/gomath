#include "../info.h"
#include "../internal/half/half.hpp"
#include "exception.hpp"
#include "fp16.h"
#include "fp16_common.hpp"
#include "tensor.h"
#include <exception>
#include <stdio.h>

#if !(defined(__APPLE__) && defined(__arm64__))

template <typename T, typename T2> class fp16_impl : public fp16 {
public:
  // dot_vector
  virtual uint16_t dot_vector(const uint16_t *x, const uint16_t *w, int64_t d) {
    size_t np = (d & ~(bs - 1));
    T y_vec = _fp16_zero_ps<T>();
    for (size_t i = 0; i < np; i += bs) {
      T x_vec = _fp16_loadu<T>(x);
      T w_vec = _fp16_loadu<T>(w);
      y_vec = _fp16_fmadd_ps<T>(x_vec, w_vec, y_vec);
      x += bs;
      w += bs;
    }
    float y = _fp16_sum_ps<T>(y_vec);
    for (size_t i = 0; i < d - np; i++) {
      y += fp16_to_fp32(x[i]) * fp16_to_fp32(w[i]);
    }
    return fp32_to_fp16(y);
  }

  virtual void mul_vector(const uint16_t *x, const uint16_t *w, uint16_t *y,
                          int64_t d) {
    size_t np = (d & ~(bs - 1));
    for (size_t i = 0; i < np; i += bs) {
      T x_vec = _fp16_loadu<T>(x);
      T w_vec = _fp16_loadu<T>(w);
      T2 y_vec = _fp16_mul_ps<T, T2>(x_vec, w_vec);
      _fp16_store_ps(y, y_vec);
      x += bs;
      w += bs;
      y += bs;
    }
    for (size_t i = 0; i < d - np; i++) {
      y[i] = fp32_to_fp16(fp16_to_fp32(x[i]) * fp16_to_fp32(w[i]));
    }
  }
};

static fp16 *_fp16;

bool fp16_init() {
  if (has_avx512()) {
    _fp16 = new fp16_impl<__m512, __m256i>();
    return true;
  } else if (has_avx()) {
    _fp16 = new fp16_impl<__m256, __m128i>();
    return true;
  }
  return false;
}

uint16_t fp16_dot_vector(const uint16_t *x, const uint16_t *w, int64_t d) {
  return _fp16->dot_vector(x, w, d);
}

void fp16_mul_vector(const uint16_t *x, const uint16_t *w, uint16_t *y,
                     int64_t d) {
  _fp16->mul_vector(x, w, y, d);
}

#endif