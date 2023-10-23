#include "tensor.h"

void tensor_init() {
  extern void bf16_init();
  extern void f32_init();
  bf16_init();
  f32_init();
}