#include "tensor.h"
#include "../info.h"
#include "fp16.h"
#include "fp32.h"

size_t bs;

bool tensor_init() {
  if (has_avx512()) {
    bs = 16;
  } else if (has_avx()) {
    bs = 8;
  }
  if (!fp16_init()) {
    return false;
  }
  if (!fp32_init()) {
    return false;
  }
  return true;
}