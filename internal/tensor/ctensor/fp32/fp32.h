#ifndef __TENSOR_F32_H__
#define __TENSOR_F32_H__

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stdint.h>

bool fp32_init();

// dot
float fp32_dot_vector(const float *x, const float *w, int64_t d);

// mul
void fp32_mul_scalar(const float x, const float *w, float *y, int64_t d);
void fp32_mul_vector(const float *x, const float *w, float *y, int64_t d);

// div
void fp32_div_scalar(const float x, const float *w, float *y, int64_t d);
void fp32_div_vector(const float *x, const float *w, float *y, int64_t d);

// add
void fp32_add_scalar(const float x, const float *w, float *y, int64_t d);
void fp32_add_vector(const float *x, const float *w, float *y, int64_t d);

#ifdef __cplusplus
}
#endif

#endif