#include "bf16.h"
#include "../info.h"
#include "exception.hpp"
#include <stdio.h>

typedef uint16_t (*fn_bf16_dot_vector)(uint16_t *, uint16_t *, int64_t);
fn_bf16_dot_vector f_bf16_dot_vector = nullptr;

void bf16_init() {
  if (has_avx512()) {
    extern uint16_t bf16_dot_vector_avx512(uint16_t *, uint16_t *, int64_t);
    f_bf16_dot_vector = bf16_dot_vector_avx512;
  } else if (has_avx2()) {
    extern uint16_t bf16_dot_vector_avx2(uint16_t *, uint16_t *, int64_t);
    f_bf16_dot_vector = bf16_dot_vector_avx2;
  } else if (has_avx()) {
    extern uint16_t bf16_dot_vector_avx(uint16_t *, uint16_t *, int64_t);
    f_bf16_dot_vector = bf16_dot_vector_avx;
  }
}

uint16_t bf16_dot_vector(uint16_t *x, uint16_t *w, int64_t d) {
  return f_bf16_dot_vector(x, w, d);
}

uint16_t bf16_dot_vector_avx(uint16_t *x, uint16_t *w, int64_t d) {
  throw not_implemented_exception("bf16_dot_vector_avx");
  return 0;
}

uint16_t bf16_dot_vector_avx2(uint16_t *x, uint16_t *w, int64_t d) {
  throw not_implemented_exception("bf16_dot_vector_avx2");
  return 0;
}

uint16_t bf16_dot_vector_avx512(uint16_t *x, uint16_t *w, int64_t d) {
  throw not_implemented_exception("bf16_dot_vector_avx512");
  return 0;
}