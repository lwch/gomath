#include "tensor.h"

void tensor_init() {
  extern void bf16_init();
  bf16_init();
}