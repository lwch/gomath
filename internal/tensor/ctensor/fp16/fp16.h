#ifndef __CTENSOR_FP16_H__
#define __CTENSOR_FP16_H__

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stdint.h>

bool fp16_init();

// dot
uint16_t fp16_dot_vector(const uint16_t *x, const uint16_t *w, int64_t d);

// mul
void fp16_mul_scalar(const uint16_t x, const uint16_t *w, uint16_t *y,
                     int64_t d);
void fp16_mul_vector(const uint16_t *x, const uint16_t *w, uint16_t *y,
                     int64_t d);

// div
void fp16_div_scalar(const uint16_t x, const uint16_t *w, uint16_t *y,
                     int64_t d);
void fp16_div_vector(const uint16_t *x, const uint16_t *w, uint16_t *y,
                     int64_t d);

// add
void fp16_add_scalar(const uint16_t x, const uint16_t *w, uint16_t *y,
                     int64_t d);
void fp16_add_vector(const uint16_t *x, const uint16_t *w, uint16_t *y,
                     int64_t d);

#ifdef __cplusplus
}
#endif

#endif