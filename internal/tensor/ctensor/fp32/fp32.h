#ifndef __TENSOR_F32_H__
#define __TENSOR_F32_H__

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stdint.h>

bool fp32_init();

// dot
float fp32_dot(const float *x, const float *w, int64_t d);

// mul
void fp32_scalar_mul(const float x, const float *w, float *y, int64_t d);
void fp32_mul(const float *x, const float *w, float *y, int64_t d);

// div
void fp32_scalar_div(const float x, const float *w, float *y, int64_t d);
void fp32_div(const float *x, const float *w, float *y, int64_t d);

// add
void fp32_scalar_add(const float x, const float *w, float *y, int64_t d);
void fp32_add(const float *x, const float *w, float *y, int64_t d);

// sub
void fp32_scalar_sub(const float x, const float *w, float *y, int64_t d);
void fp32_sub(const float *x, const float *w, float *y, int64_t d);

#ifdef __cplusplus
}
#endif

#endif