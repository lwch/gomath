#ifndef __CTENSOR_FP16_H__
#define __CTENSOR_FP16_H__

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stdint.h>

bool fp16_init();

// dot
uint16_t fp16_dot(const uint16_t *x, const uint16_t *w, int64_t d);

// mul
void fp16_scalar_mul(const uint16_t x, const uint16_t *w, uint16_t *y,
                     int64_t d);
void fp16_mul(const uint16_t *x, const uint16_t *w, uint16_t *y, int64_t d);

// div
void fp16_scalar_div(const uint16_t x, const uint16_t *w, uint16_t *y,
                     int64_t d);
void fp16_div(const uint16_t *x, const uint16_t *w, uint16_t *y, int64_t d);

// add
void fp16_scalar_add(const uint16_t x, const uint16_t *w, uint16_t *y,
                     int64_t d);
void fp16_add(const uint16_t *x, const uint16_t *w, uint16_t *y, int64_t d);

// sub
void fp16_scalar_sub(const uint16_t x, const uint16_t *w, uint16_t *y,
                     int64_t d);
void fp16_sub(const uint16_t *x, const uint16_t *w, uint16_t *y, int64_t d);

// pow
void fp16_pow(const uint16_t *x, const uint16_t n, uint16_t *y, int64_t d);

#ifdef __cplusplus
}
#endif

#endif